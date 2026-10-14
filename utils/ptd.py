# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host-side Progressive Tree Drafting (PTD) core.

Model-agnostic (pure python/numpy) implementation of the PTD decoding state:
a width/depth-bounded persistent draft tree, a request-local candidate pool,
per-round packing of tree rows into a fixed slot budget, and greedy
verification with accepted-path extraction.

Semantics follow the paper (arXiv 2607.10661, Section 3 and Appendix D) and
the torq-tools PTD implementation this module was ported from; the per-round
token/position bookkeeping:

- Committed KV cache holds L rows (tokens x_0..x_{L-1}).
- Slot 0 is the anchor: the last emitted (predicted) token, at position L;
  its KV row is committed by the model at row L.
- Packed tree nodes sit at slots 1..N-1; a node at depth d (first level = 1)
  gets logical position L + d (siblings share a position). A node sees the
  full committed context (keys <= L) plus its own same-tree ancestors only.
- Greedy verification: the anchor prediction must equal a packed candidate
  root to start the walk; the walk follows argmax predictions that match
  *packed* children; the argmax of the last visited node is always emitted
  as the terminal bonus (no KV yet -- it is the next round's anchor).

Algorithmic attribution
-----------------------
This module is a domain-specific adaptation of Progressive Tree Drafting
(PTD), a training-free speculative decoding strategy:

- Paper: "Unlocking Parallelism in Autoregressive Language Models via
  Speculative Decoding with Progressive Tree Drafting" (COLM 2026).
  Zipeng Gao, Zhi Zheng, Qingrong Xia, Junda Lin, Ziwei Zhao, Tong Xu,
  Zhefeng Wang, Enhong Chen -- University of Science and Technology of
  China (MINE-USTC) and Huawei Technologies.
  https://arxiv.org/abs/2607.10661
- Reference implementation (algorithm semantics only; nothing vendored):
  https://github.com/MINE-USTC/PTD

Domain-specific adaptations (vs. the paper/reference):
1. Fixed row budget: live tree nodes are packed into a bounded number of
   fixed slots per forward, with host-supplied per-row positions and a
   visibility matrix (static-graph constraint), instead of the reference's
   dynamic per-step batching.
2. Scratch/committed KV split: tree-row K/V is never persisted; the host
   promotes the accepted path into the committed cache (static-cache model
   constraint), instead of the reference's DynamicCache updates.
3. Greedy verification only (the paper's sampling walk is not implemented).
4. Simplified tree maintenance: FIFO eviction on width overflow, stepping
   that promotes the earliest non-duplicate child, re-rooting on the
   round's committed token, a request-local pool keyed by the single last
   emitted token, and pool updates from packed draft nodes only.
5. Optional prompt/history suffix lookup supplies additional candidate chains
   from observed tokens; each proposal is still verified by the target model.

The fixed-row PTD body this core drives (the standalone ``--ptd-rows``
export of the torq pipeline, compiled to ``transformer.vmfb`` together with
``lm_head.vmfb``) is the model for every phase: it takes per-row tokens,
per-row logical positions, and a float32 0/1 visibility matrix, and returns
per-row hidden states or logits plus the committed cache (anchor blended in
at position L) and per-slot combined key/value scratch rows the host
promotes on accepted paths. Prompt prefill packs causal chains the same way
as draft trees.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

StopCheck = Callable[[int, Sequence[int]], bool]


def lookup_continuation(
    tokens: Sequence[int], budget: int, max_ngram: int = 4, min_ngram: int = 1
) -> list[int]:
    """Propose an earlier continuation of the longest matching token suffix.

    Prefer the most recent match of equal length. Proposals contain only
    observed tokens; the target model must still verify the entire path.
    """
    if budget <= 0:
        return []
    history = list(tokens)
    for size in range(min(max_ngram, len(history) - 1), max(min_ngram, 1) - 1, -1):
        suffix = history[-size:]
        for start in range(len(history) - size - 1, -1, -1):
            if history[start:start + size] == suffix:
                return history[start + size:start + size + budget]
    return []


@dataclass(eq=False)
class PTDNode:
    """One token node in a draft or candidate tree (identity-hashed)."""

    value: int
    depth: int
    parent: "PTDNode | None" = None
    # Insertion-ordered children (oldest first); FIFO eviction and stepping
    # both rely on this order.
    children: "OrderedDict[int, PTDNode]" = field(default_factory=OrderedDict)
    # Hops to the deepest leaf (0 for a leaf); maintained by the owning tree.
    child_depth: int = 0


class PTDTree:
    """A token tree with a virtual root; first-level nodes are depth 1.

    Width: each node keeps at most ``width`` children (FIFO eviction of the
    oldest child when a new one is added). Depth: nodes deeper than
    ``max_depth`` do not grow; a first-level subtree that exceeds
    ``max_depth`` is "stepped" (its earliest-added child is promoted to the
    first level, per the paper's stepping mechanism) via :meth:`step_and_prune`.
    """

    def __init__(self, width: int = 4, max_depth: int = 6):
        if width < 1:
            raise ValueError("width must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.width = width
        self.max_depth = max_depth
        self.root = PTDNode(value=-1, depth=0)

    @property
    def node_count(self) -> int:
        return len(self.bfs_nodes())

    def bfs_nodes(self) -> list[PTDNode]:
        """All nodes, breadth-first, insertion order within each level."""
        nodes: list[PTDNode] = []
        queue: deque[PTDNode] = deque(self.root.children.values())
        while queue:
            node = queue.popleft()
            nodes.append(node)
            queue.extend(node.children.values())
        return nodes

    def subtree_depth(self, node: PTDNode) -> int:
        """Total depth of the subtree rooted at ``node`` (node included)."""
        return node.depth + node.child_depth

    def _refresh_order(self, parent: PTDNode, value: int) -> None:
        child = parent.children.pop(value)
        parent.children[value] = child

    def add_child(self, parent: PTDNode, value: int) -> PTDNode | None:
        """Add ``value`` as a child of ``parent``.

        Dedup: an existing child retains its insertion order.
        Growth is allowed one level past ``max_depth`` (the transient that
        :meth:`step_and_prune` resolves); deeper parents do not grow.
        """
        if parent.depth + 1 > self.max_depth + 1:
            return None
        if value in parent.children:
            return parent.children[value]
        if len(parent.children) >= self.width:
            evicted = parent.children.pop(next(iter(parent.children)))
            evicted.parent = None
        child = PTDNode(value=value, depth=parent.depth + 1, parent=parent)
        parent.children[value] = child
        self._recompute_depths(parent)
        return child

    def add_first_level(self, value: int) -> PTDNode | None:
        return self.add_child(self.root, value)

    def _recompute_depths(self, node: PTDNode) -> None:
        """Recompute child_depth up the ancestor chain from ``node``."""
        current = node
        while current is not None:
            current.child_depth = 0 if not current.children else 1 + max(
                child.child_depth for child in current.children.values()
            )
            current = current.parent

    def _fix_subtree_depths(self, node: PTDNode) -> None:
        """Re-depth a re-parented subtree (depth follows the parent chain)."""
        node.depth = node.parent.depth + 1
        for child in node.children.values():
            self._fix_subtree_depths(child)

    def step_and_prune(self) -> None:
        """Paper stepping: for each first-level subtree deeper than
        ``max_depth``, promote its earliest-added child to the first level;
        drop the subtree when no such child exists. 
        Promoted nodes take the newest first-level position."""
        for value in list(self.root.children):
            node = self.root.children.get(value)
            while node is not None and self.subtree_depth(node) > self.max_depth:
                first_level_values = set(self.root.children)
                promoted: PTDNode | None = None
                for grandchild in node.children.values():
                    if grandchild.value in first_level_values:
                        continue
                    promoted = grandchild
                    break
                if promoted is None:
                    del self.root.children[node.value]
                    node.parent = None
                    node = None
                    break
                # Promote: node is replaced by its earliest non-duplicate
                # child; the child's siblings (stale branches) are pruned.
                del self.root.children[node.value]
                for sibling in node.children.values():
                    if sibling is not promoted:
                        sibling.parent = None
                node.children = OrderedDict()
                node.parent = None
                promoted.parent = self.root
                self._fix_subtree_depths(promoted)
                self.root.children[promoted.value] = promoted
                node = promoted
        self._recompute_depths(self.root)

    def reroot(self, value: int) -> None:
        """Re-anchor the tree after a round: keep the first level bounded
        (FIFO eviction) and add/refresh ``value`` as a first-level node."""
        while len(self.root.children) > self.width:
            evicted = self.root.children.pop(next(iter(self.root.children)))
            evicted.parent = None
        if value in self.root.children:
            self._refresh_order(self.root, value)
        else:
            self.add_first_level(value)


class PTDPool:
    """Request-local candidate pool: token -> tree of continuations.

    Merging a draft node into the pool applies the M(T, T') at the
    node's value: the node's subtree (what follows that token) is merged
    into the entry keyed by the token. Entries are bounded in node count
    (oldest first-level child evicted) and the pool in entry count
    (oldest entry evicted, insertion/LRU order).
    """

    def __init__(self, max_entry_nodes: int = 16, max_entries: int = 64, width: int = 8):
        self.max_entry_nodes = max_entry_nodes
        self.max_entries = max_entries
        self.width = width
        self._entries: "OrderedDict[int, PTDTree]" = OrderedDict()

    def merge_node(self, node: PTDNode) -> None:
        if not node.children:
            return
        entry = self._entries.get(node.value)
        if entry is None:
            entry = PTDTree(width=self.width, max_depth=self.max_entry_nodes)
            self._entries[node.value] = entry
        self._merge_subtree(entry, node)
        self._entries.move_to_end(node.value)
        self._prune_entry(node.value)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def _merge_subtree(self, entry: PTDTree, node: PTDNode) -> None:
        """Merge ``node``'s children structure into ``entry`` (paper M)."""
        for child in list(node.children.values()):
            target = entry.add_child(entry.root, child.value)
            if target is not None:
                self._merge_subtree_children(entry, target, child)

    @staticmethod
    def _merge_subtree_children(entry: PTDTree, target: PTDNode, source: PTDNode) -> None:
        for child in source.children.values():
            next_target = entry.add_child(target, child.value)
            if next_target is None:
                continue
            PTDPool._merge_subtree_children(entry, next_target, child)

    def _prune_entry(self, key: int) -> None:
        entry = self._entries[key]
        while entry.node_count > self.max_entry_nodes and entry.root.children:
            oldest = next(iter(entry.root.children))
            del entry.root.children[oldest]
        if entry.node_count == 0:
            del self._entries[key]

    def retrieve(self, key: int) -> PTDTree | None:
        """The candidate tree of continuations after ``key``, or None."""
        entry = self._entries.get(key)
        if entry is None or entry.node_count == 0:
            return None
        self._entries.move_to_end(key)
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_nodes(self) -> int:
        return sum(entry.node_count for entry in self._entries.values())


@dataclass
class PtdPacking:
    """One round's row layout for the fixed-N PTD body."""

    tokens: np.ndarray      # [N] int64; slot 0 = anchor
    positions: np.ndarray   # [N] int64; per-row logical positions
    vis: np.ndarray         # [N, N-1] float32; row i sees scratch slot j+1 iff set
    candidate_slots: dict   # PTDNode -> slot
    draft_slots: dict       # PTDNode -> slot
    n_candidates: int
    n_drafts: int


def pack_round(
    L: int,
    anchor_token: int,
    candidate_nodes: Sequence[PTDNode],
    draft_nodes: Sequence[PTDNode],
    rows: int,
    pad_token: int,
) -> PtdPacking:
    """Lay out a round: slot 0 = anchor, then candidate rows, then draft rows;
    leftover slots are invalid (pad token, position L, zero visibility).

    ``candidate_nodes`` / ``draft_nodes`` must already be budget-capped, in
    the order they should occupy slots (BFS per tree).
    """
    tokens = np.full(rows, pad_token, dtype=np.int64)
    positions = np.full(rows, L, dtype=np.int64)
    vis = np.zeros((rows, rows - 1), dtype=np.float32)
    tokens[0] = anchor_token

    candidate_slots: dict[PTDNode, int] = {}
    draft_slots: dict[PTDNode, int] = {}
    cursor = 1
    for nodes, slots in ((candidate_nodes, candidate_slots), (draft_nodes, draft_slots)):
        for node in nodes:
            if cursor >= rows:
                break
            slots[node] = cursor
            tokens[cursor] = node.value
            positions[cursor] = L + node.depth
            cursor += 1

    # Visibility: a row sees its own slot plus its same-tree ancestors.
    for slots in (candidate_slots, draft_slots):
        for node, row in slots.items():
            chain = node
            while chain is not None:
                ancestor_slot = slots.get(chain)
                if ancestor_slot is not None:
                    vis[row, ancestor_slot - 1] = 1
                chain = chain.parent

    return PtdPacking(
        tokens=tokens,
        positions=positions,
        vis=vis,
        candidate_slots=candidate_slots,
        draft_slots=draft_slots,
        n_candidates=len(candidate_slots),
        n_drafts=len(draft_slots),
    )


@dataclass
class PtdRoundResult:
    emitted: list          # committed output tokens of the round (stop-truncated)
    # PTDNode path whose KVs are promoted; accepted == emitted[:-1] as tokens.
    # Committed cache length after the round: L' = L + 1 + len(accepted)
    # (anchor row + accepted input nodes); the final emitted token (bonus) has
    # no KV yet and is the next round's anchor at position L'.
    accepted: list
    stopped: bool          # stop condition triggered within the emission
    hit: bool              # candidate root matched the anchor prediction
    bonus: int             # terminal emitted prediction (next round's anchor)


def greedy_verify(
    row_tokens: np.ndarray,
    candidate_nodes: Sequence[PTDNode],
    candidate_slots: dict,
    gen_tail: Sequence[int],
    stop_check: StopCheck,
    max_accepted: int,
) -> PtdRoundResult:
    """Greedy verification of one round (paper Algorithm 1, steps 5-6).

    ``row_tokens`` contains one greedy prediction per packed row, in model
    token IDs (already remapped if the vocabulary is trimmed). Slot 0 is
    the anchor prediction. ``candidate_nodes`` must be exactly the packed
    candidate nodes (in slot order) and ``candidate_slots`` maps each to its
    slot index. ``max_accepted`` bounds how many accepted input nodes fit in
    the cache (budget cap); the emission is truncated at the first token
    triggering ``stop_check(token, tokens_committed_before_it)``.
    """
    if max_accepted < 0:
        raise ValueError("max_accepted must be >= 0")
    anchor_pred = int(row_tokens[0])
    roots = [n for n in candidate_nodes if n.depth == 1]
    root = next((r for r in roots if r.value == anchor_pred), None)
    accepted: list[PTDNode] = []
    bonus = anchor_pred
    node = root
    while node is not None and len(accepted) < max_accepted:
        accepted.append(node)
        bonus = int(row_tokens[candidate_slots[node]])
        child = node.children.get(bonus)
        node = child if child in candidate_slots else None
    emitted = [node.value for node in accepted] + [bonus]

    # Stop scan: ordinary greedy evaluates the stop condition after every
    # emitted token, so the multi-token emission must be scanned the same
    # way (tokens committed before ``t`` = gen_tail + preceding emissions).
    stopped = False
    for i, token in enumerate(emitted):
        if stop_check(token, list(gen_tail) + emitted[:i]):
            emitted = emitted[: i + 1]
            stopped = True
            break
    return PtdRoundResult(
        emitted=emitted,
        accepted=accepted[: len(emitted) - 1],
        stopped=stopped,
        hit=root is not None,
        bonus=emitted[-1],
    )
