# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The Piper voices this demo ships, and what differs between them.

A voice is a complete set of assets — partA, the partB window vmfbs, and the
config carrying its phoneme->id map — plus the espeak language used to
phonemize for it. The espeak dictionaries are shared: one ``espeak-ng-data``
covers every language, so adding a voice adds no phonemizer assets.

Voices also differ in *shape*: a multi-speaker VITS export takes a speaker id
and conditions the vocoder on a speaker embedding, while a single-speaker one
has neither. Nothing here encodes that — :mod:`piper_core.pipeline` reads the
signatures off the models themselves — but ``speakers`` lets the CLI reject an
out-of-range ``--speaker`` before anything loads. Likewise the sample rate
(22.05 kHz for medium voices, 16 kHz for low) is read from the voice config.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Voice:
    """One installable voice: its assets, its espeak language, its samples."""

    key: str
    language: str
    espeak: str
    speakers: int
    samples: tuple[str, ...]
    subdir: str = ""      # asset subdir under the model dir ("" = top level)

    @property
    def config_name(self) -> str:
        return f"{self.key}.onnx.json"

    def asset(self, *parts: str) -> str:
        """Path of an asset for this voice, relative to the model dir."""
        return "/".join((self.subdir, *parts)) if self.subdir else "/".join(parts)


EN_US_LIBRITTS_R: Final = Voice(
    key="en_US-libritts_r-medium",
    language="English (US)",
    espeak="en-us",
    speakers=904,
    samples=(
        "The morning train was late again. Nobody on the platform seemed surprised.",
        "Rain fell steadily on the harbour road. The ferry would not sail until morning, "
        "and the lamps along the quay came on one by one.",
        "She opened the wooden gate and crossed the wet grass. Below the cliff, the grey "
        "water moved slowly against the rocks.",
        "The bakery on the corner opens at six. By seven the shelves are half empty, and "
        "by nine there is nothing left but rye.",
        "At midnight the lighthouse changed its rhythm. Three short flashes, then a long "
        "pause, exactly as the old keeper had promised.",
    ),
)

# A low-quality (16 kHz) single-speaker voice: a lighter, faster alternative to
# the default. Same samples as the default, so the two are directly comparable.
EN_US_LESSAC_LOW: Final = Voice(
    key="en_US-lessac-low",
    language="English (US), 16 kHz",
    espeak="en-us",
    speakers=1,
    subdir="en_US-lessac-low",
    samples=EN_US_LIBRITTS_R.samples,
)

# Latin American Spanish. The samples mirror the English ones scene for scene,
# so the two voices can be compared directly.
ES_MX_ALD: Final = Voice(
    key="es_MX-ald-medium",
    language="Spanish (Mexico)",
    espeak="es-419",
    speakers=1,
    subdir="es_MX-ald-medium",
    samples=(
        "El tren de la mañana llegó tarde otra vez. Nadie en el andén pareció sorprenderse.",
        "La lluvia caía sin parar sobre el camino del puerto. El transbordador no zarparía "
        "hasta la mañana, y las luces del muelle se encendieron una por una.",
        "Abrió la puerta de madera y cruzó la hierba mojada. Bajo el acantilado, el agua "
        "gris se movía despacio contra las rocas.",
        "La panadería de la esquina abre a las seis. Para las siete los estantes están medio "
        "vacíos, y para las nueve no queda más que pan de centeno.",
        "A medianoche el faro cambió su ritmo. Tres destellos cortos, luego una pausa larga, "
        "exactamente como lo había prometido el viejo farero.",
    ),
)

VOICES: Final[dict[str, Voice]] = {v.key: v for v in (EN_US_LIBRITTS_R, EN_US_LESSAC_LOW, ES_MX_ALD)}
DEFAULT_VOICE: Final[str] = EN_US_LIBRITTS_R.key


def get_voice(key: str | None = None) -> Voice:
    """Look up a voice by key; ``None`` gives the default (English)."""
    if key is None:
        return VOICES[DEFAULT_VOICE]
    try:
        return VOICES[key]
    except KeyError:
        raise ValueError(f"unknown voice {key!r}; choose from {', '.join(VOICES)}") from None
