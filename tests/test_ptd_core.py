# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Unit tests for the PTD host core (tree, pool, packing, verification).

Mirrors the torq-tools PTD core tests: utils/ptd.py is the host core
for the standalone fixed-row PTD body."""

import numpy as np
import pytest

from utils.ptd import (
    PTDTree,
    PTDPool,
    greedy_verify,
    lookup_continuation,
    pack_round,
)


@pytest.mark.parametrize("tokens,budget,ngram,expected", [
    ([], 4, 4, []),
    ([1], 4, 4, []),
    ([1, 2, 3], 4, 4, []),
    ([1, 2, 3, 4, 1, 2], 2, 4, [3, 4]),
    ([1, 2, 3, 4, 1, 2], 0, 4, []),
    ([1, 2, 3, 4, 1, 2], 4, 0, []),
    ([1, 2, 8, 2, 9, 1, 2], 1, 4, [8]),
    ([1, 2, 8, 2, 9, 1, 2], 1, 1, [9]),
    ([1, 2, 1, 3, 1], 2, 4, [3, 1]),
])
def test_lookup_continuation(tokens, budget, ngram, expected):
    assert lookup_continuation(tokens, budget, ngram) == expected


def test_lookup_minimum_match_length():
    assert lookup_continuation([2, 1, 2], 4, min_ngram=2) == []
    assert lookup_continuation([2, 1, 2], 4) == [1, 2]


# ---------------------------------------------------------------------------
# Draft tree
# ---------------------------------------------------------------------------

def _build_chain(tree, values):
    """Add a root->v1->v2... chain; returns the leaf node."""
    node = None
    for value in values:
        node = tree.add_child(tree.root if node is None else node, value)
    return node


def test_bfs_order_is_level_then_insertion():
    tree = PTDTree(width=4, max_depth=4)
    a = tree.add_child(tree.root, 10)
    b = tree.add_child(tree.root, 20)
    a1 = tree.add_child(a, 101)
    b1 = tree.add_child(b, 201)
    tree.add_child(a1, 1011)
    values = [n.value for n in tree.bfs_nodes()]
    assert values == [10, 20, 101, 201, 1011]
    assert tree.node_count == 5
    assert tree.subtree_depth(a) == 3
    assert tree.subtree_depth(b1) == 2


def test_add_child_dedup_preserves_order():
    tree = PTDTree(width=4, max_depth=4)
    a = tree.add_child(tree.root, 10)
    tree.add_child(tree.root, 20)
    tree.add_child(tree.root, 30)
    again = tree.add_child(tree.root, 10)
    assert again is a
    assert list(tree.root.children) == [10, 20, 30]
    assert tree.node_count == 3


def test_width_fifo_eviction_drops_oldest_subtree():
    tree = PTDTree(width=2, max_depth=4)
    a = tree.add_child(tree.root, 10)
    tree.add_child(a, 101)
    tree.add_child(tree.root, 20)
    tree.add_child(tree.root, 30)  # evicts 10 (and its child 101)
    assert list(tree.root.children) == [20, 30]
    assert tree.node_count == 2


def test_growth_capped_one_past_max_depth():
    tree = PTDTree(width=4, max_depth=2)
    a = tree.add_child(tree.root, 10)      # depth 1
    a1 = tree.add_child(a, 11)             # depth 2
    a11 = tree.add_child(a1, 12)           # depth 3 (transient, allowed)
    assert a11 is not None
    assert tree.add_child(a11, 13) is None  # depth 4 (blocked)
    tree.step_and_prune()
    assert all(tree.subtree_depth(n) <= 2 for n in tree.root.children.values())


def test_stepping_promotes_earliest_child():
    tree = PTDTree(width=4, max_depth=2)
    a = tree.add_child(tree.root, 10)
    a1 = tree.add_child(a, 11)
    tree.add_child(a1, 12)  # subtree depth 3 > 2
    assert tree.subtree_depth(a) == 3
    tree.step_and_prune()
    # first level now holds 11 (earliest child of 10) with child 12
    assert list(tree.root.children) == [11]
    assert tree.root.children[11].value == 11
    assert tree.root.children[11].children[12].value == 12
    assert tree.subtree_depth(tree.root.children[11]) == 2


def test_stepping_skips_first_level_duplicate_values():
    tree = PTDTree(width=4, max_depth=2)
    a = tree.add_child(tree.root, 10)
    a1 = tree.add_child(a, 11)
    tree.add_child(a1, 12)  # deep subtree under 10
    tree.add_child(tree.root, 11)  # 11 already at first level
    tree.step_and_prune()
    # earliest child 11 is a first-level duplicate -> subtree dropped
    assert list(tree.root.children) == [11]
    assert 12 not in tree.root.children[11].children


def test_stepping_preserves_earliest_child_after_duplicate_prediction():
    tree = PTDTree(max_depth=2)
    root = tree.add_first_level(10)
    earliest = tree.add_child(root, 11)
    tree.add_child(root, 20)
    tree.add_child(root, 11)
    tree.add_child(earliest, 12)
    tree.step_and_prune()
    assert list(tree.root.children) == [11]


def test_reroot_evicts_and_adds():
    tree = PTDTree(width=2, max_depth=4)
    tree.add_child(tree.root, 10)
    tree.add_child(tree.root, 20)
    tree.reroot(20)   # refresh only
    assert list(tree.root.children) == [10, 20]
    tree.reroot(30)   # evict oldest (10), add 30
    assert list(tree.root.children) == [20, 30]


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def test_pool_merge_and_retrieve_structure():
    tree = PTDTree(width=4, max_depth=4)
    a = tree.add_child(tree.root, 10)
    tree.add_child(a, 11)
    tree.add_child(a, 12)
    a11 = tree.add_child(a, 11)
    tree.add_child(a11, 111)

    pool = PTDPool(max_entry_nodes=16, max_entries=8)
    for node in tree.bfs_nodes():
        pool.merge_node(node)

    entry_10 = pool.retrieve(10)
    assert entry_10 is not None
    # after 10: 11 and 12; after 11 (node a's child): 111
    assert set(entry_10.root.children) == {11, 12}
    assert set(entry_10.root.children[11].children) == {111}
    # node 11's own subtree (111) is merged under key 11 as well
    entry_11 = pool.retrieve(11)
    assert entry_11 is not None
    assert set(entry_11.root.children) == {111}
    assert pool.retrieve(999) is None


def test_pool_merge_is_idempotent():
    tree = PTDTree(width=4, max_depth=4)
    a = tree.add_child(tree.root, 10)
    tree.add_child(a, 11)
    pool = PTDPool()
    pool.merge_node(a)
    pool.merge_node(a)
    assert pool.retrieve(10).node_count == 1


def test_pool_entry_node_cap_prunes_oldest_first_level():
    pool = PTDPool(max_entry_nodes=3, max_entries=8)
    big = PTDTree(width=8, max_depth=4)
    b = big.add_child(big.root, 7)
    for v in range(10):
        big.add_child(b, v)
    pool.merge_node(b)
    entry = pool.retrieve(7)
    assert entry.node_count <= 3
    assert set(entry.root.children) == {7, 8, 9}  # oldest evicted


def test_pool_entry_count_cap_evicts_lru():
    pool = PTDPool(max_entry_nodes=16, max_entries=2)
    t1 = PTDTree(width=4, max_depth=2)
    n1 = t1.add_child(t1.root, 1)
    t1.add_child(n1, 11)
    t2 = PTDTree(width=4, max_depth=2)
    n2 = t2.add_child(t2.root, 2)
    t2.add_child(n2, 22)
    t3 = PTDTree(width=4, max_depth=2)
    n3 = t3.add_child(t3.root, 3)
    t3.add_child(n3, 33)
    pool.merge_node(n1)
    pool.merge_node(n2)
    pool.retrieve(1)  # touches key 1 (LRU refresh)
    pool.merge_node(n3)  # evicts key 2 (least recently used)
    assert pool.retrieve(1) is not None
    assert pool.retrieve(2) is None
    assert pool.retrieve(3) is not None


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def _two_node_paths():
    """candidate: 10 -> 11 ; draft: 20 -> 21, 20 -> 22."""
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    c11 = cand.add_child(c10, 11)
    draft = PTDTree(width=4, max_depth=4)
    d20 = draft.add_child(draft.root, 20)
    d21 = draft.add_child(d20, 21)
    d22 = draft.add_child(d20, 22)
    return cand, [c10, c11], draft, [d20, d21, d22]


def test_pack_round_layout_positions_and_vis():
    cand, cand_nodes, draft, draft_nodes = _two_node_paths()
    L = 5
    p = pack_round(L, anchor_token=99, candidate_nodes=cand_nodes,
                  draft_nodes=draft_nodes, rows=8, pad_token=0)
    # slot 0 = anchor; candidates first (BFS), then drafts
    assert p.tokens[0] == 99
    assert list(p.tokens[1:6]) == [10, 11, 20, 21, 22]
    assert p.positions[0] == L
    # sibling isolation: 21 and 22 share a position; 10/20 are first level
    assert p.positions[1] == L + 1   # cand depth 1
    assert p.positions[2] == L + 2   # cand depth 2
    assert p.positions[3] == L + 1   # draft depth 1
    assert p.positions[4] == L + 2   # draft depth 2 (21)
    assert p.positions[5] == L + 2   # draft depth 2 (22, sibling of 21)
    # unused slots are pad @ L
    assert list(p.tokens[6:]) == [0, 0]
    assert list(p.positions[6:]) == [L, L]

    assert p.vis.dtype == np.float32
    # vis (row, scratch-slot-1): row 2 (11) sees slot 1 (10) + itself
    assert p.vis[2, 0] == 1 and p.vis[2, 1] == 1
    # row 2 does not see draft rows (cross-tree isolation) or the pad rows
    assert (p.vis[2, 2:] == 0).all()
    # row 1 (10, first level) sees only itself
    assert p.vis[1, 0] == 1 and (p.vis[1, 1:] == 0).all()
    # row 4 (21) sees slot 3 (20) + itself, not sibling 22 (slot 5)
    assert p.vis[4, 2] == 1 and p.vis[4, 3] == 1
    assert p.vis[4, 4] == 0
    # row 5 (22) sees slot 3 (20) + itself, not sibling 21 (slot 4)
    assert p.vis[5, 2] == 1 and p.vis[5, 4] == 1 and p.vis[5, 3] == 0
    # anchor row sees no scratch rows
    assert (p.vis[0] == 0).all()
    # pad rows have zero visibility
    assert (p.vis[6] == 0).all() and (p.vis[7] == 0).all()
    assert p.n_candidates == 2 and p.n_drafts == 3
    assert p.candidate_slots[cand.root.children[10]] == 1
    assert p.draft_slots[draft.root.children[20].children[22]] == 5


def test_pack_round_truncates_to_budget():
    cand, cand_nodes, draft, draft_nodes = _two_node_paths()
    p = pack_round(5, 99, cand_nodes, draft_nodes, rows=4, pad_token=0)
    # slots: anchor + 2 candidates + 1 draft (21); 22 dropped
    assert p.n_candidates == 2 and p.n_drafts == 1
    assert 22 not in [n.value for n in p.draft_slots]
    assert list(p.tokens) == [99, 10, 11, 20]


# ---------------------------------------------------------------------------
# Greedy verification
# ---------------------------------------------------------------------------

def _row_tokens(tokens_by_row):
    return np.array([tokens_by_row[i] for i in range(len(tokens_by_row))], dtype=np.int64)


def _no_stop(token, committed):
    return False


def test_verify_no_candidates_emits_anchor_pred():
    result = greedy_verify(_row_tokens({0: 7}), [], {}, [], _no_stop, max_accepted=8)
    assert result.emitted == [7]
    assert result.accepted == []
    assert result.bonus == 7
    assert result.hit is False
    assert result.stopped is False


def test_verify_root_mismatch_emits_anchor_pred():
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    cand.add_child(c10, 11)
    slots = {c10: 1, cand.root.children[10].children[11]: 2}
    # anchor predicts 12; candidate root is 10 -> no hit
    result = greedy_verify(_row_tokens({0: 12, 1: 21, 2: 22}),
                           cand.bfs_nodes(), slots, [], _no_stop, max_accepted=8)
    assert result.emitted == [12]
    assert result.accepted == []
    assert result.hit is False


def test_verify_full_hit_with_bonus():
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    c101 = cand.add_child(c10, 11)
    slots = {c10: 1, c101: 2}
    # anchor -> 10 (matches root); node 10 verifies -> 11 (matches child);
    # node 11 verifies -> 12 (bonus, no matching child)
    result = greedy_verify(_row_tokens({0: 10, 1: 11, 2: 12}),
                           cand.bfs_nodes(), slots, [], _no_stop, max_accepted=8)
    assert result.emitted == [10, 11, 12]
    assert result.accepted == [c10, c101]
    assert result.bonus == 12
    assert result.hit is True


def test_verify_partial_hit_stops_at_unpacked_child():
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    c101 = cand.add_child(c10, 11)
    cand.add_child(c101, 12)  # 12 exists but has no row
    # only 10 and 11 packed (12 has no row)
    slots = {c10: 1, c101: 2}
    result = greedy_verify(_row_tokens({0: 10, 1: 11, 2: 12}),
                           cand.bfs_nodes(), slots, [], _no_stop, max_accepted=8)
    # walk stops at 11 (its verify token 12 is an unpacked child);
    # bonus = 11's verify token
    assert result.emitted == [10, 11, 12]
    assert result.accepted == [c10, c101]


def test_verify_partial_hit_stops_at_mismatch():
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    cand.add_child(c10, 11)
    slots = {c10: 1, cand.root.children[10].children[11]: 2}
    # node 10 verifies to 13 (not a child) -> stop, bonus 13
    result = greedy_verify(_row_tokens({0: 10, 1: 13, 2: 14}),
                           cand.bfs_nodes(), slots, [], _no_stop, max_accepted=8)
    assert result.emitted == [10, 13]
    assert result.accepted == [c10]


def test_verify_stop_scan_truncates_mid_emission():
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    c101 = cand.add_child(c10, 11)
    slots = {c10: 1, c101: 2}

    def stop_at_eos(token, committed):
        return token == 11  # 11 acts as EOS

    result = greedy_verify(_row_tokens({0: 10, 1: 11, 2: 12}),
                           cand.bfs_nodes(), slots, [], stop_at_eos, max_accepted=8)
    # emitted [10, 11, 12] truncated at 11 (inclusive)
    assert result.emitted == [10, 11]
    assert result.accepted == [c10]
    assert result.stopped is True


def test_verify_stop_scan_uses_committed_tail_per_token():
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    c101 = cand.add_child(c10, 11)
    slots = {c10: 1, c101: 2}
    # double-newline shaped rule: stop when the committed tail ends [5, 10]
    # and the new token is 11 (mirrors: committed [-2:] pattern + next token)
    seen = []

    def tail_rule(token, committed):
        seen.append((token, list(committed)))
        return len(committed) >= 2 and committed[-2:] == [5, 10] and token == 11

    result = greedy_verify(_row_tokens({0: 10, 1: 11, 2: 12}),
                           cand.bfs_nodes(), slots, [1, 2, 5, 5], tail_rule, max_accepted=8)
    # token 10 (committed tail [5,5]) does not trigger; token 11 (tail [5,10]) does
    assert result.stopped is True
    assert result.emitted == [10, 11]
    assert seen[0] == (10, [1, 2, 5, 5])
    assert seen[1] == (11, [1, 2, 5, 5, 10])


def test_verify_budget_cap_recomputes_bonus_at_cut():
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    c101 = cand.add_child(c10, 11)
    c1012 = cand.add_child(c101, 12)
    cand.add_child(c1012, 13)
    slots = {c10: 1, c101: 2, c1012: 3, next(iter(c1012.children.values())): 4}
    result = greedy_verify(_row_tokens({0: 10, 1: 11, 2: 12, 3: 13, 4: 14}),
                           cand.bfs_nodes(), slots, [], _no_stop, max_accepted=2)
    # full walk would accept 10,11,12,13 + bonus 14; cap at 2 accepted
    assert result.accepted == [c10, c101]
    assert result.emitted == [10, 11, 12]  # bonus = node 11's verify token
    assert result.bonus == 12
    assert result.stopped is False


@pytest.mark.parametrize("budget, expected, stopped", [(0, [10], False), (1, [10, 11], False), (2, [10, 11, 12], True)])
def test_verify_budget_applies_before_stop_scan(budget, expected, stopped):
    cand = PTDTree()
    first = cand.add_first_level(10)
    second = cand.add_child(first, 11)
    seen = []

    def stop(token, committed):
        seen.append(token)
        return token == 12

    result = greedy_verify(np.array([10, 11, 12]), [first, second],
                           {first: 1, second: 2}, [], stop, max_accepted=budget)
    assert result.emitted == expected
    assert len(result.accepted) <= budget
    assert result.stopped is stopped
    assert seen == expected


def test_verify_rejects_negative_budget():
    with pytest.raises(ValueError, match="max_accepted"):
        greedy_verify(np.array([10]), [], {}, [], _no_stop, max_accepted=-1)
