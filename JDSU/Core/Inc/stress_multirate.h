#ifndef __STRESS_MULTIRATE_H
#define __STRESS_MULTIRATE_H

#include <stdint.h>
#include "stress_table.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The calibrated 45-point stress table is still replayed in full on every
 * frame.  This scheduler decides only which CH1 ADC values are refreshed.
 * This preserves DAC row order and the hidden predecessor, but skipping ADC
 * waits DOES change dwell-time history and may change optical response.
 * Static row certification alone does not certify reduced-profile behavior.
 *
 * Multirate acquisition is deliberately opt-in.  An old host never sends the
 * arm command, so every point remains fresh and its frame is indistinguishable
 * from the established MAP45 acquisition.
 */

#define STRESS_MULTIRATE_CHANNEL_MASK             0x02U
#define STRESS_MULTIRATE_FRESH_BITMAP_BYTES       \
    ((STRESS_TABLE_POINT_COUNT + 7U) / 8U)
#define STRESS_MULTIRATE_SAMPLE_TIME_UNAVAILABLE  0xFFFFU
#define STRESS_MULTIRATE_SAMPLE_TIME_UNIT_US      2U
#define STRESS_MULTIRATE_SAMPLE_TIME_MAX_US       \
    (0xFFFEUL * STRESS_MULTIRATE_SAMPLE_TIME_UNIT_US)
#define STRESS_MULTIRATE_CONTINUOUS_MAP_PERIOD    0U
#define STRESS_MULTIRATE_REFERENCE_MAP_PERIOD     1U
#define STRESS_MULTIRATE_DEFAULT_MAP_PERIOD       30U
#define STRESS_MULTIRATE_MIN_MAP_PERIOD           3U
#define STRESS_MULTIRATE_MAX_MAP_PERIOD           100U
#define STRESS_MULTIRATE_DEFAULT_ACTIVATE_CODES   40U
#define STRESS_MULTIRATE_DEFAULT_RELEASE_CODES    20U
#define STRESS_MULTIRATE_DEFAULT_RELEASE_FRAMES   4U
#define STRESS_MULTIRATE_BASELINE_QUIET_FRAMES    4U
#define STRESS_MULTIRATE_BASELINE_EMA_SHIFT       6U
#define STRESS_MULTIRATE_STARTUP_STABLE_FRAMES    8U
#define STRESS_MULTIRATE_STARTUP_MAX_FRAMES       32U
#define STRESS_MULTIRATE_MAX_MAP_DEFERRAL_FRAMES  120U
#define STRESS_MULTIRATE_SURVEY_FRESH_COUNT       STRESS_SEGMENT_COUNT
#define STRESS_MULTIRATE_TRACK_WIDE_FRESH_COUNT   \
    (STRESS_SEGMENT_COUNT + 4U)
#define STRESS_MULTIRATE_TRACK_SINGLE_FRESH_COUNT \
    (STRESS_SEGMENT_COUNT + 2U)
/* Compatibility name used by the v3 host contract. */
#define STRESS_MULTIRATE_TRACK_FRESH_COUNT        \
    STRESS_MULTIRATE_TRACK_WIDE_FRESH_COUNT
#define STRESS_MULTIRATE_SINGLE_CONFIRM_FRAMES    8U
#define STRESS_MULTIRATE_ADC_MAX_CODE              4095U
#define STRESS_MULTIRATE_MAP_AGE_MASK              0x7FU
#define STRESS_MULTIRATE_BANDWIDTH_GAP_FLAG        0x80U
#define STRESS_MULTIRATE_NO_SEGMENT                0xFFU

/* mapPeriodFrames == 0 still performs the compulsory first MAP45, then
 * disables every automatic MAP for uninterrupted dynamic acquisition.
 * mapPeriodFrames == 1 is an explicit complete-spectrum reference session:
 * every frame samples all 45 CH1 rows and carries the bandwidth-gap flag.
 * It is not a settled single-value teacher or a 15 Hz pressure measurement. */

typedef enum
{
    STRESS_MULTIRATE_PROFILE_MAP = 0U,
    STRESS_MULTIRATE_PROFILE_SURVEY = 1U,
    STRESS_MULTIRATE_PROFILE_TRACK_WIDE = 2U,
    /* Compatibility name retained for the v3 TRACK13 profile. */
    STRESS_MULTIRATE_PROFILE_TRACK = STRESS_MULTIRATE_PROFILE_TRACK_WIDE,
    STRESS_MULTIRATE_PROFILE_TRACK_SINGLE = 3U
} StressMultirateProfile_t;

/* The compulsory MAP45 is acquired through a different optical history than
 * the following sparse loop.  LEARNING explicitly suppresses contact
 * detection while a bounded set of SURVEY9 frames learns that sparse-loop
 * steady state.  FORCED_READY means the hard bound expired before eight
 * consecutive stable frames; detection is nevertheless live from the next
 * frame so startup learning can never mask a later contact indefinitely. */
typedef enum
{
    STRESS_MULTIRATE_BASELINE_READY = 0U,
    STRESS_MULTIRATE_BASELINE_LEARNING = 1U,
    STRESS_MULTIRATE_BASELINE_FORCED_READY = 2U
} StressMultirateBaselineStatus_t;

typedef struct
{
    uint8_t profile;
    uint8_t primarySegment;
    uint8_t secondarySegment;
    uint8_t freshCount;
    uint8_t mapAgeFrames;
    uint8_t bandwidthDiscontinuity;
    uint8_t freshBitmap[STRESS_MULTIRATE_FRESH_BITMAP_BYTES];
    uint32_t planSequence;
} StressMultiratePlan_t;

typedef struct
{
    uint8_t enabled;
    uint8_t mapPeriodFrames;
    uint16_t activateThresholdCodes;
    uint16_t releaseThresholdCodes;
    uint8_t releaseFrames;
    /* Negotiated session capability.  Configure defaults this to zero and
     * StartSession deliberately preserves it, so a v3 host can never receive
     * the v4-only profile 3. */
    uint8_t singleTrackEnabled;
    /* Explicit v5 experiment: G8 watches right flank (38), not left (36).
     * Configure clears it. Never change it in an active scan session. */
    uint8_t g8RightSentinel;

    uint8_t cacheValid[STRESS_MULTIRATE_FRESH_BITMAP_BYTES];
    uint16_t cachedCh1[STRESS_TABLE_POINT_COUNT];
    uint16_t baselineCh1[STRESS_TABLE_POINT_COUNT];
    uint32_t baselineAccumulatorQ6[STRESS_TABLE_POINT_COUNT];
    uint8_t baselineValid;
    uint8_t baselineQuietFrames;
    uint8_t startupBaselineFrames;
    uint8_t startupBaselineStableFrames;
    uint8_t baselineStatus;

    uint16_t frameSegmentScore[STRESS_SEGMENT_COUNT];
    uint16_t lastSegmentScore[STRESS_SEGMENT_COUNT];
    uint16_t frameSentinelSeenMask;
    uint8_t frameSentinelOutOfDomain;
    uint8_t activityLatched;
    uint8_t activePrimarySegment;
    uint8_t activeSecondarySegment;
    uint8_t singleTrackActive;
    uint8_t compactUniqueFrames;
    uint8_t compactCandidateSegment;
    uint8_t lowActivityFrames;
    uint8_t framesSinceMap;
    uint8_t forceMapRequested;
    uint32_t nextPlanSequence;
    StressMultiratePlan_t currentPlan;
} StressMultirateState_t;

/* Configure does not begin a scan session.  StartSession always clears the
 * cache/baseline and therefore forces the first frame to MAP45. */
void StressMultirate_Configure(StressMultirateState_t *state,
                              uint8_t enabled,
                              uint8_t mapPeriodFrames,
                              uint16_t activateThresholdCodes,
                              uint16_t releaseThresholdCodes);
void StressMultirate_SetSingleTrackEnabled(StressMultirateState_t *state,
                                           uint8_t enabled);
void StressMultirate_SetG8RightSentinel(StressMultirateState_t *state,
                                       uint8_t enabled);
void StressMultirate_StartSession(StressMultirateState_t *state);
void StressMultirate_ForceMap(StressMultirateState_t *state);

/* BeginFrame returns an immutable plan until EndFrame. */
const StressMultiratePlan_t *StressMultirate_BeginFrame(
        StressMultirateState_t *state);
uint8_t StressMultirate_ShouldSample(
        const StressMultiratePlan_t *plan, uint16_t pointIndex);
void StressMultirate_RecordCh1(StressMultirateState_t *state,
                              uint16_t pointIndex,
                              uint16_t adcCode);
uint8_t StressMultirate_GetCachedCh1(const StressMultirateState_t *state,
                                    uint16_t pointIndex,
                                    uint16_t *adcCode);
void StressMultirate_EndFrame(StressMultirateState_t *state);

uint8_t StressMultirate_IsEnabled(const StressMultirateState_t *state);
uint8_t StressMultirate_AllCached(const StressMultirateState_t *state);
const StressMultiratePlan_t *StressMultirate_CurrentPlan(
        const StressMultirateState_t *state);
StressMultirateBaselineStatus_t StressMultirate_BaselineStatus(
        const StressMultirateState_t *state);

#ifdef __cplusplus
}
#endif

#endif
