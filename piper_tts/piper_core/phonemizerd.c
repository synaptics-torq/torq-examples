/* phonemizerd — persistent espeak-ng phonemizer for the TTS menu.
 *
 * Loads espeak once, then serves requests on stdin: one line of text in,
 * one "SEG\t<sentence_end 0|1>\t<ipa phonemes + punctuation>" line per
 * espeak segment out, terminated by "DONE". Mirrors libpiper's
 * piper_synthesize_start() phonemization exactly (same API call, same
 * terminator classification, same punctuation append).
 *
 * Build (on the board):
 *   I=$HOME/piper-demo/piper1-gpl/libpiper/build/espeak_ng-install
 *   U=$HOME/piper-demo/piper1-gpl/libpiper/build/espeak_ng/src/espeak_ng_external-build/src/ucd-tools
 *   gcc -O2 phonemizerd.c -I$I/include $I/lib/libespeak-ng.a $U/libucd.a -lm -lpthread -o phonemizerd
 *
 * Run: ./phonemizerd <espeak-data-dir> [voice]
 */
#include <espeak-ng/speak_lib.h>
#include <stdio.h>

/* CLAUSE_* are piper-side definitions (piper_impl.hpp), not in speak_lib.h */
#define CLAUSE_INTONATION_FULL_STOP 0x00000000
#define CLAUSE_INTONATION_COMMA 0x00001000
#define CLAUSE_INTONATION_QUESTION 0x00002000
#define CLAUSE_INTONATION_EXCLAMATION 0x00003000
#define CLAUSE_TYPE_CLAUSE 0x00040000
#define CLAUSE_TYPE_SENTENCE 0x00080000
#define CLAUSE_PERIOD (40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COMMA (20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_QUESTION (40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_EXCLAMATION (45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COLON (30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_SEMICOLON (30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <espeak-data-dir> [voice]\n", argv[0]);
        return 1;
    }
    const char *voice = argc > 2 ? argv[2] : "en-us";
    if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, argv[1], 0) < 0) {
        fprintf(stderr, "espeak_Initialize failed\n");
        return 1;
    }
    if (espeak_SetVoiceByName(voice) != EE_OK) {
        fprintf(stderr, "espeak_SetVoiceByName(%s) failed\n", voice);
        return 1;
    }
    printf("READY\n");
    fflush(stdout);

    char line[8192];
    while (fgets(line, sizeof line, stdin)) {
        size_t n = strlen(line);
        while (n && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
        if (!n) { printf("DONE\n"); fflush(stdout); continue; }

        const void *ptr = line;
        while (ptr != NULL) {
            int terminator = 0;
            const char *ph = espeak_TextToPhonemesWithTerminator(
                &ptr, espeakCHARS_AUTO, espeakPHONEMES_IPA, &terminator);

            /* piper.cpp: terminator &= 0x000FFFFF, then classify */
            int t = terminator & 0x000FFFFF;
            const char *punct = "";
            if      (t == CLAUSE_PERIOD)      punct = ".";
            else if (t == CLAUSE_QUESTION)    punct = "?";
            else if (t == CLAUSE_EXCLAMATION) punct = "!";
            else if (t == CLAUSE_COMMA)       punct = ", ";
            else if (t == CLAUSE_COLON)       punct = ": ";
            else if (t == CLAUSE_SEMICOLON)   punct = "; ";
            int sentence_end =
                ((terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE);

            printf("SEG\t%d\t%s%s\n", sentence_end, ph ? ph : "", punct);
        }
        printf("DONE\n");
        fflush(stdout);
    }
    return 0;
}
