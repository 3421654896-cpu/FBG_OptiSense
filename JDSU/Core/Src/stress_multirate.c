#include "stress_multirate.h"

#include <string.h>

#if (STRESS_SEGMENT_COUNT == 0U)
#error "stress multirate requires at least one FBG segment"
#endif

#if ((STRESS_TABLE_POINT_COUNT % STRESS_SEGMENT_COUNT) != 0U)
#error "stress multirate requires equal contiguous points per FBG segment"
#endif

#define STRESS_POINTS_PER_SEGMENT \
    (STRESS_TABLE_POINT_COUNT / STRESS_SEGMENT_COUNT)
#define STRESS_SENTINEL_ALL_MASK \
    ((uint16_t)((1UL << STRESS_SEGMENT_COUNT) - 1UL))

#if (STRESS_POINTS_PER_SEGMENT < 3U)
#error "stress multirate needs left/centre/right samples for every segment"
#endif

#if (STRESS_SEGMENT_COUNT != 9U) || (STRESS_POINTS_PER_SEGMENT != 5U)
#error "adaptive TRACK11/TRACK13 require the certified 9 x 5 stress table"
#endif

#if (STRESS_MULTIRATE_MAX_MAP_DEFERRAL_FRAMES > STRESS_MULTIRATE_MAP_AGE_MASK)
#error "maximum MAP deferral must fit the v3 seven-bit map-age field"
#endif

#if (STRESS_MULTIRATE_MAX_MAP_DEFERRAL_FRAMES <= STRESS_MULTIRATE_MAX_MAP_PERIOD)
#error "active MAP deferral must exceed every routine MAP period"
#endif

/* Audited from
 * Python/artifacts/ch1_measured_template_selection_v4_certified_20260905.json
 * SHA-256 2330F3D5091744329F0170B8823BDE89D3BDDCB5897DACE67C64A7113C8E9C3E.
 * For each grating, offsets 1 and 3 both have reliability weight 1.0; this
 * table selects the one with the greater absolute
 * selected_template_slope_v_per_nm.  All nine are used by SURVEY9; the seven
 * non-ROI reads keep every grating observable during TRACK11/TRACK13.
 */
static const uint8_t trackSentinelLocalOffset[STRESS_SEGMENT_COUNT] =
{
    3U, 3U, 3U, 3U, 1U, 3U, 1U, 1U, 1U
};

static uint8_t bitIsSet(const uint8_t *bitmap, uint16_t index)
{
    return (bitmap[index >> 3U] & (uint8_t)(1U << (index & 7U))) ? 1U : 0U;
}

static void setBit(uint8_t *bitmap, uint16_t index)
{
    bitmap[index >> 3U] |= (uint8_t)(1U << (index & 7U));
}

static uint16_t absoluteDifference(uint16_t left, uint16_t right)
{
    return (left >= right) ? (left - right) : (right - left);
}

static uint16_t segmentBase(uint8_t segment)
{
    return (uint16_t)segment * (uint16_t)STRESS_POINTS_PER_SEGMENT;
}

static uint16_t segmentLeftFlank(uint8_t segment)
{
    /* With the certified five points this is offset one.  The formula also
     * keeps a future odd, uniformly grouped table away from its endpoints. */
    uint16_t offset = (uint16_t)STRESS_POINTS_PER_SEGMENT / 4U;
    if(offset == 0U) offset = 1U;
    return segmentBase(segment) + offset;
}

static uint16_t segmentCentre(uint8_t segment)
{
    return segmentBase(segment) + (uint16_t)STRESS_POINTS_PER_SEGMENT / 2U;
}

static uint16_t segmentRightFlank(uint8_t segment)
{
    uint16_t leftOffset = segmentLeftFlank(0U);
    return segmentBase(segment) + (uint16_t)STRESS_POINTS_PER_SEGMENT
            - 1U - leftOffset;
}

static uint16_t segmentSentinel(const StressMultirateState_t *state, uint8_t segment)
{
    if(state->g8RightSentinel && segment == 7U) return segmentBase(segment) + 3U;
    return segmentBase(segment) + trackSentinelLocalOffset[segment];
}

static void clearPlan(StressMultiratePlan_t *plan)
{
    uint32_t sequence = plan->planSequence;
    memset(plan, 0, sizeof(*plan));
    plan->planSequence = sequence;
    plan->primarySegment = STRESS_MULTIRATE_NO_SEGMENT;
    plan->secondarySegment = STRESS_MULTIRATE_NO_SEGMENT;
}

static void addFreshPoint(StressMultiratePlan_t *plan, uint16_t pointIndex)
{
    if(pointIndex >= STRESS_TABLE_POINT_COUNT ||
       bitIsSet(plan->freshBitmap, pointIndex)) return;
    setBit(plan->freshBitmap, pointIndex);
    plan->freshCount++;
}

static void buildMapPlan(StressMultiratePlan_t *plan)
{
    plan->profile = STRESS_MULTIRATE_PROFILE_MAP;
    for(uint16_t point = 0U; point < STRESS_TABLE_POINT_COUNT; point++)
        addFreshPoint(plan, point);
}

static void buildSurveyPlan(const StressMultirateState_t *state, StressMultiratePlan_t *plan)
{
    plan->profile = STRESS_MULTIRATE_PROFILE_SURVEY;
    for(uint8_t segment = 0U; segment < STRESS_SEGMENT_COUNT; segment++)
    {
        /* SURVEY9 is a no-contact/full-space watcher.  Its one certified
         * high-slope sentinel per grating provides >30 Hz nominal discovery;
         * it deliberately does not claim a two-flank peak-centre estimate. */
        addFreshPoint(plan, segmentSentinel(state, segment));
    }
}

/* Q8 inverse absolute slopes of the certified sentinel templates (V/nm).
 * The common TIA scale cancels when ranking channels.  These are sensitivity
 * weights, not a learned noise model or a calibrated displacement estimate. */
static const uint16_t sentinelInverseSlopeQ8[STRESS_SEGMENT_COUNT] =
{136U, 79U, 82U, 205U, 128U, 134U, 182U, 71U, 177U};

static uint32_t rankScore(const StressMultirateState_t *state, uint8_t segment)
{
    /* Same template certificate as above: abs(left slope)=3.49747321285,
     * abs(right)=1.50701508697 V/nm. Preserve the old ranking scale:
     * round(71 * 3.49747321285 / 1.50701508697) = 165.
     * This is static sensitivity weighting, NOT transient correction. */
    uint16_t weight = (state->g8RightSentinel && segment == 7U)
            ? 165U : sentinelInverseSlopeQ8[segment];
    return (uint32_t)state->lastSegmentScore[segment] *
            (state->singleTrackEnabled ? weight : 1U);
}

static void twoLargestSegments(const StressMultirateState_t *state,
                               uint8_t *primary,
                               uint8_t *secondary)
{
    uint8_t first = 0U;
    uint8_t second = (STRESS_SEGMENT_COUNT > 1U) ? 1U : 0U;
    if(rankScore(state, second) > rankScore(state, first))
    {
        uint8_t swap = first;
        first = second;
        second = swap;
    }
    for(uint8_t segment = 2U; segment < STRESS_SEGMENT_COUNT; segment++)
    {
        if(rankScore(state, segment) > rankScore(state, first))
        {
            second = first;
            first = segment;
        }
        else if(rankScore(state, segment) > rankScore(state, second))
        {
            second = segment;
        }
    }
    *primary = first;
    *secondary = second;
}

static void buildTrackWidePlan(const StressMultirateState_t *state,
                               StressMultiratePlan_t *plan)
{
    uint8_t primary = state->activePrimarySegment;
    uint8_t secondary = state->activeSecondarySegment;
    plan->profile = STRESS_MULTIRATE_PROFILE_TRACK_WIDE;
    plan->primarySegment = primary;
    plan->secondarySegment = secondary;

    /* Three inner points retain sign/curvature information for each of the
     * two strongest spatial responses in the conservative wide profile. */
    addFreshPoint(plan, segmentLeftFlank(primary));
    addFreshPoint(plan, segmentCentre(primary));
    addFreshPoint(plan, segmentRightFlank(primary));
    addFreshPoint(plan, segmentLeftFlank(secondary));
    addFreshPoint(plan, segmentCentre(secondary));
    addFreshPoint(plan, segmentRightFlank(secondary));

    /* One high-slope sentinel for every other grating makes this TRACK13,
     * preserving full spatial observability at the dynamic frame rate. */
    for(uint8_t segment = 0U; segment < STRESS_SEGMENT_COUNT; segment++)
    {
        if(segment == primary || segment == secondary) continue;
        addFreshPoint(plan, segmentSentinel(state, segment));
    }
}

static void buildTrackSinglePlan(const StressMultirateState_t *state,
                                 StressMultiratePlan_t *plan)
{
    uint8_t primary = state->activePrimarySegment;
    plan->profile = STRESS_MULTIRATE_PROFILE_TRACK_SINGLE;
    plan->primarySegment = primary;
    plan->secondarySegment = STRESS_MULTIRATE_NO_SEGMENT;

    /* The primary keeps its inner triplet.  Other gratings keep the common
     * certified sentinels; their full peak shape is NOT fresh in this frame. */
    addFreshPoint(plan, segmentLeftFlank(primary));
    addFreshPoint(plan, segmentCentre(primary));
    addFreshPoint(plan, segmentRightFlank(primary));
    for(uint8_t segment = 0U; segment < STRESS_SEGMENT_COUNT; segment++)
    {
        if(segment == primary) continue;
        addFreshPoint(plan, segmentSentinel(state, segment));
    }
}

static void buildTrackPlan(const StressMultirateState_t *state,
                           StressMultiratePlan_t *plan)
{
    if(state->singleTrackEnabled && state->singleTrackActive)
        buildTrackSinglePlan(state, plan);
    else
        buildTrackWidePlan(state, plan);
}

static uint8_t sentinelsCompleteAndInDomain(
        const StressMultirateState_t *state)
{
    return (state->frameSentinelSeenMask == STRESS_SENTINEL_ALL_MASK &&
            !state->frameSentinelOutOfDomain) ? 1U : 0U;
}

static uint8_t compactUniqueResponse(const StressMultirateState_t *state,
                                     uint8_t primary,
                                     uint8_t secondary)
{
    uint32_t sum = 0U;
    uint64_t sumSquares = 0U;
    uint32_t primaryScore;
    uint32_t secondaryScore;

    if(!sentinelsCompleteAndInDomain(state) ||
       primary >= STRESS_SEGMENT_COUNT || secondary >= STRESS_SEGMENT_COUNT)
        return 0U;

    primaryScore = rankScore(state, primary);
    secondaryScore = rankScore(state, secondary);
    if(state->lastSegmentScore[primary] < state->activateThresholdCodes)
        return 0U;

    for(uint8_t segment = 0U; segment < STRESS_SEGMENT_COUNT; segment++)
    {
        uint32_t score = rankScore(state, segment);
        if(segment != primary &&
           state->lastSegmentScore[segment] >= state->activateThresholdCodes)
            return 0U;
        sum += score;
        sumSquares += (uint64_t)score * (uint64_t)score;
    }
    if(sum == 0U) return 0U;

    /* A sentinel alone cannot qualify an unseen ROI as a good single peak.
     * Require this frame's primary triplet and a resolvable interior rise.
     * Detailed template residual/OOD rejection remains a host responsibility. */
    {
        uint16_t left = segmentLeftFlank(primary);
        uint16_t centre = segmentCentre(primary);
        uint16_t right = segmentRightFlank(primary);
        uint16_t minimum;
        if(!bitIsSet(state->currentPlan.freshBitmap, left) ||
           !bitIsSet(state->currentPlan.freshBitmap, centre) ||
           !bitIsSet(state->currentPlan.freshBitmap, right)) return 0U;
        if(state->cachedCh1[left] >= STRESS_MULTIRATE_ADC_MAX_CODE ||
           state->cachedCh1[centre] >= STRESS_MULTIRATE_ADC_MAX_CODE ||
           state->cachedCh1[right] >= STRESS_MULTIRATE_ADC_MAX_CODE) return 0U;
        minimum = state->cachedCh1[left] < state->cachedCh1[right]
                ? state->cachedCh1[left] : state->cachedCh1[right];
        if((uint32_t)state->cachedCh1[centre] <=
           (uint32_t)minimum + state->releaseThresholdCodes) return 0U;
    }

    /* Integer-exact forms of the conservative v4 gates:
     * primary/sum >= 0.60; second/primary < 0.35; and
     * N_eff=(sum^2/sumSquares) <= 1.8. */
    if((uint32_t)primaryScore * 5UL < sum * 3UL) return 0U;
    if((uint32_t)secondaryScore * 20UL >=
       (uint32_t)primaryScore * 7UL) return 0U;
    if((uint64_t)5U * (uint64_t)sum * (uint64_t)sum >
       (uint64_t)9U * sumSquares) return 0U;
    return 1U;
}

void StressMultirate_Configure(StressMultirateState_t *state,
                              uint8_t enabled,
                              uint8_t mapPeriodFrames,
                              uint16_t activateThresholdCodes,
                              uint16_t releaseThresholdCodes)
{
    if(state == 0) return;
    memset(state, 0, sizeof(*state));
    state->enabled = enabled ? 1U : 0U;
    if(mapPeriodFrames != STRESS_MULTIRATE_CONTINUOUS_MAP_PERIOD &&
       mapPeriodFrames != STRESS_MULTIRATE_REFERENCE_MAP_PERIOD &&
       mapPeriodFrames < STRESS_MULTIRATE_MIN_MAP_PERIOD)
        mapPeriodFrames = STRESS_MULTIRATE_DEFAULT_MAP_PERIOD;
    if(mapPeriodFrames > STRESS_MULTIRATE_MAX_MAP_PERIOD)
        mapPeriodFrames = STRESS_MULTIRATE_MAX_MAP_PERIOD;
    state->mapPeriodFrames = mapPeriodFrames;
    state->activateThresholdCodes = activateThresholdCodes
            ? activateThresholdCodes : STRESS_MULTIRATE_DEFAULT_ACTIVATE_CODES;
    state->releaseThresholdCodes = releaseThresholdCodes
            ? releaseThresholdCodes : STRESS_MULTIRATE_DEFAULT_RELEASE_CODES;
    if(state->releaseThresholdCodes > state->activateThresholdCodes)
        state->releaseThresholdCodes = state->activateThresholdCodes;
    state->releaseFrames = STRESS_MULTIRATE_DEFAULT_RELEASE_FRAMES;
    state->baselineStatus = STRESS_MULTIRATE_BASELINE_LEARNING;
    state->currentPlan.primarySegment = STRESS_MULTIRATE_NO_SEGMENT;
    state->currentPlan.secondarySegment = STRESS_MULTIRATE_NO_SEGMENT;
}

void StressMultirate_SetSingleTrackEnabled(StressMultirateState_t *state,
                                           uint8_t enabled)
{
    if(state == 0) return;
    state->singleTrackEnabled = enabled ? 1U : 0U;
    if(!state->singleTrackEnabled)
    {
        state->singleTrackActive = 0U;
        state->compactUniqueFrames = 0U;
        state->compactCandidateSegment = STRESS_MULTIRATE_NO_SEGMENT;
    }
}

void StressMultirate_SetG8RightSentinel(StressMultirateState_t *state,
                                       uint8_t enabled)
{
    if(state == 0) return;
    state->g8RightSentinel = enabled ? 1U : 0U;
}

void StressMultirate_StartSession(StressMultirateState_t *state)
{
    if(state == 0) return;
    memset(state->cacheValid, 0, sizeof(state->cacheValid));
    memset(state->cachedCh1, 0, sizeof(state->cachedCh1));
    memset(state->baselineCh1, 0, sizeof(state->baselineCh1));
    memset(state->baselineAccumulatorQ6, 0,
           sizeof(state->baselineAccumulatorQ6));
    memset(state->frameSegmentScore, 0, sizeof(state->frameSegmentScore));
    memset(state->lastSegmentScore, 0, sizeof(state->lastSegmentScore));
    state->frameSentinelSeenMask = 0U;
    state->frameSentinelOutOfDomain = 0U;
    state->baselineValid = 0U;
    state->baselineQuietFrames = 0U;
    state->startupBaselineFrames = 0U;
    state->startupBaselineStableFrames = 0U;
    state->baselineStatus = STRESS_MULTIRATE_BASELINE_LEARNING;
    state->activityLatched = 0U;
    state->activePrimarySegment = STRESS_MULTIRATE_NO_SEGMENT;
    state->activeSecondarySegment = STRESS_MULTIRATE_NO_SEGMENT;
    state->singleTrackActive = 0U;
    state->compactUniqueFrames = 0U;
    state->compactCandidateSegment = STRESS_MULTIRATE_NO_SEGMENT;
    state->lowActivityFrames = 0U;
    state->framesSinceMap = state->mapPeriodFrames;
    state->forceMapRequested = 0U;
    state->nextPlanSequence = 0U;
    memset(&state->currentPlan, 0, sizeof(state->currentPlan));
    state->currentPlan.primarySegment = STRESS_MULTIRATE_NO_SEGMENT;
    state->currentPlan.secondarySegment = STRESS_MULTIRATE_NO_SEGMENT;
}

void StressMultirate_ForceMap(StressMultirateState_t *state)
{
    if(state == 0) return;
    state->forceMapRequested = 1U;
}

uint8_t StressMultirate_AllCached(const StressMultirateState_t *state)
{
    if(state == 0) return 0U;
    for(uint16_t point = 0U; point < STRESS_TABLE_POINT_COUNT; point++)
        if(!bitIsSet(state->cacheValid, point)) return 0U;
    return 1U;
}

const StressMultiratePlan_t *StressMultirate_BeginFrame(
        StressMultirateState_t *state)
{
    StressMultiratePlan_t *plan;
    if(state == 0) return 0;
    plan = &state->currentPlan;
    plan->planSequence = state->nextPlanSequence++;
    clearPlan(plan);
    plan->mapAgeFrames = (state->framesSinceMap >
            STRESS_MULTIRATE_MAP_AGE_MASK)
            ? STRESS_MULTIRATE_MAP_AGE_MASK : state->framesSinceMap;

    if(state->enabled &&
       state->mapPeriodFrames == STRESS_MULTIRATE_REFERENCE_MAP_PERIOD)
    {
        /* Explicit reference acquisition must never wait for a quiet signal
         * or switch to cached sparse rows during contact. Keep CH1-only IO,
         * all DAC safety/path constraints and per-point timing unchanged.
         * Every such frame, including startup, is excluded from bandwidth
         * acceptance by the existing on-wire gap flag. */
        buildMapPlan(plan);
        plan->bandwidthDiscontinuity = 1U;
    }
    else if(!state->enabled || !StressMultirate_AllCached(state))
    {
        buildMapPlan(plan);
    }
    else if(state->forceMapRequested)
    {
        buildMapPlan(plan);
        plan->bandwidthDiscontinuity = state->activityLatched ? 1U : 0U;
    }
    else if(state->activityLatched)
    {
        if(state->mapPeriodFrames != STRESS_MULTIRATE_CONTINUOUS_MAP_PERIOD &&
           state->framesSinceMap >=
           (uint8_t)(STRESS_MULTIRATE_MAX_MAP_DEFERRAL_FRAMES - 1U))
        {
            /* A bounded shape-check protects against indefinite spectral
             * drift.  It cannot satisfy the 15 Hz gap while MAP45 runs, so
             * v3 marks this frame explicitly instead of claiming continuity. */
            buildMapPlan(plan);
            plan->bandwidthDiscontinuity = 1U;
        }
        else buildTrackPlan(state, plan);
    }
    else if(state->mapPeriodFrames != STRESS_MULTIRATE_CONTINUOUS_MAP_PERIOD &&
            state->framesSinceMap >=
            (uint8_t)(state->mapPeriodFrames - 1U) &&
            state->baselineQuietFrames >=
            STRESS_MULTIRATE_BASELINE_QUIET_FRAMES)
    {
        /* Routine MAP45 is deferred until the unloaded signal has remained
         * quiet.  This keeps it out of the active 15 Hz measurement window. */
        buildMapPlan(plan);
    }
    else
    {
        buildSurveyPlan(state, plan);
    }
    state->forceMapRequested = 0U;
    memset(state->frameSegmentScore, 0, sizeof(state->frameSegmentScore));
    state->frameSentinelSeenMask = 0U;
    state->frameSentinelOutOfDomain = 0U;
    return plan;
}

uint8_t StressMultirate_ShouldSample(
        const StressMultiratePlan_t *plan, uint16_t pointIndex)
{
    if(plan == 0 || pointIndex >= STRESS_TABLE_POINT_COUNT) return 0U;
    return bitIsSet(plan->freshBitmap, pointIndex);
}

void StressMultirate_RecordCh1(StressMultirateState_t *state,
                              uint16_t pointIndex,
                              uint16_t adcCode)
{
    uint8_t segment;
    uint16_t difference;
    if(state == 0 || pointIndex >= STRESS_TABLE_POINT_COUNT) return;
    state->cachedCh1[pointIndex] = adcCode;
    setBit(state->cacheValid, pointIndex);
    if(!state->baselineValid) return;
    segment = (uint8_t)(pointIndex / STRESS_POINTS_PER_SEGMENT);
    if(pointIndex != segmentSentinel(state, segment)) return;

    /* Cross-segment activity is ranked only from the nine common certified
     * sentinels.  The two extra samples in a tracked ROI are for peak-shape
     * estimation and must never give that ROI three chances to win. */
    state->frameSentinelSeenMask |= (uint16_t)(1U << segment);
    if(adcCode > STRESS_MULTIRATE_ADC_MAX_CODE)
        state->frameSentinelOutOfDomain = 1U;
    difference = absoluteDifference(adcCode, state->baselineCh1[pointIndex]);
    state->frameSegmentScore[segment] = difference;
}

uint8_t StressMultirate_GetCachedCh1(const StressMultirateState_t *state,
                                    uint16_t pointIndex,
                                    uint16_t *adcCode)
{
    if(state == 0 || adcCode == 0 || pointIndex >= STRESS_TABLE_POINT_COUNT ||
       !bitIsSet(state->cacheValid, pointIndex)) return 0U;
    *adcCode = state->cachedCh1[pointIndex];
    return 1U;
}

void StressMultirate_EndFrame(StressMultirateState_t *state)
{
    uint16_t maximum = 0U;
    uint8_t primary;
    uint8_t secondary;
    if(state == 0) return;

    if(state->currentPlan.profile == STRESS_MULTIRATE_PROFILE_MAP)
    {
        state->framesSinceMap = 0U;
        if(!state->baselineValid && StressMultirate_AllCached(state))
        {
            memcpy(state->baselineCh1, state->cachedCh1,
                   sizeof(state->baselineCh1));
            for(uint16_t point = 0U; point < STRESS_TABLE_POINT_COUNT; point++)
            {
                state->baselineAccumulatorQ6[point] =
                        (uint32_t)state->baselineCh1[point] <<
                        STRESS_MULTIRATE_BASELINE_EMA_SHIFT;
            }
            state->baselineValid = 1U;
            state->baselineQuietFrames = 0U;
            state->startupBaselineFrames = 0U;
            state->startupBaselineStableFrames = 0U;
            state->baselineStatus = STRESS_MULTIRATE_BASELINE_LEARNING;
            memset(state->lastSegmentScore, 0,
                   sizeof(state->lastSegmentScore));
            return;
        }
    }
    else if(state->framesSinceMap < 0xFFU)
    {
        state->framesSinceMap++;
    }

    /* MAP45 and SURVEY9 deliberately replay the same certified 45-row DAC
     * path, but the first sparse loop can still start from a different laser
     * and ADC steady state.  Treating the MAP value as an immediately armed
     * contact baseline caused a static system to latch TRACK13 on its first
     * sentinel frame.  Learn only the nine fresh SURVEY sentinels here,
     * replacing each baseline with the latest sparse-loop observation.  Eight
     * consecutive frame-to-frame changes inside the release band finish
     * learning.  Eight stable frames reject the short quiet pockets observed
     * inside the real CH1/G4 laser settling transient; 32 frames is still a
     * finite hard upper bound (about 0.95 s at the measured 33.7 Hz SURVEY
     * rate).  Detection begins on the following frame, so a later real contact
     * is never absorbed by this startup grace period.
     *
     * An explicit force-MAP during learning restarts the bounded SURVEY
     * qualification from that new complete baseline. */
    if(state->baselineValid &&
       state->baselineStatus == STRESS_MULTIRATE_BASELINE_LEARNING)
    {
        if(state->currentPlan.profile == STRESS_MULTIRATE_PROFILE_MAP)
        {
            memcpy(state->baselineCh1, state->cachedCh1,
                   sizeof(state->baselineCh1));
            for(uint16_t point = 0U; point < STRESS_TABLE_POINT_COUNT; point++)
            {
                state->baselineAccumulatorQ6[point] =
                        (uint32_t)state->baselineCh1[point] <<
                        STRESS_MULTIRATE_BASELINE_EMA_SHIFT;
            }
            state->startupBaselineFrames = 0U;
            state->startupBaselineStableFrames = 0U;
        }
        else
        {
            for(uint8_t segment = 0U; segment < STRESS_SEGMENT_COUNT; segment++)
                if(state->frameSegmentScore[segment] > maximum)
                    maximum = state->frameSegmentScore[segment];

            if(state->startupBaselineFrames < 0xFFU)
                state->startupBaselineFrames++;
            if(sentinelsCompleteAndInDomain(state) &&
               maximum <= state->releaseThresholdCodes)
            {
                if(state->startupBaselineStableFrames < 0xFFU)
                    state->startupBaselineStableFrames++;
            }
            else
            {
                state->startupBaselineStableFrames = 0U;
            }

            for(uint16_t point = 0U; point < STRESS_TABLE_POINT_COUNT; point++)
            {
                if(!bitIsSet(state->currentPlan.freshBitmap, point)) continue;
                state->baselineCh1[point] = state->cachedCh1[point];
                state->baselineAccumulatorQ6[point] =
                        (uint32_t)state->cachedCh1[point] <<
                        STRESS_MULTIRATE_BASELINE_EMA_SHIFT;
            }

            if(state->startupBaselineStableFrames >=
               STRESS_MULTIRATE_STARTUP_STABLE_FRAMES)
                state->baselineStatus = STRESS_MULTIRATE_BASELINE_READY;
            else if(state->startupBaselineFrames >=
                    STRESS_MULTIRATE_STARTUP_MAX_FRAMES)
                state->baselineStatus =
                        STRESS_MULTIRATE_BASELINE_FORCED_READY;
        }

        state->baselineQuietFrames = 0U;
        state->activityLatched = 0U;
        state->activePrimarySegment = STRESS_MULTIRATE_NO_SEGMENT;
        state->activeSecondarySegment = STRESS_MULTIRATE_NO_SEGMENT;
        state->singleTrackActive = 0U;
        state->compactUniqueFrames = 0U;
        state->compactCandidateSegment = STRESS_MULTIRATE_NO_SEGMENT;
        state->lowActivityFrames = 0U;
        memset(state->lastSegmentScore, 0,
               sizeof(state->lastSegmentScore));
        return;
    }

    if(sentinelsCompleteAndInDomain(state))
    {
        uint8_t wasLatched = state->activityLatched;
        uint8_t compact;
        memcpy(state->lastSegmentScore, state->frameSegmentScore,
               sizeof(state->lastSegmentScore));
        for(uint8_t segment = 0U; segment < STRESS_SEGMENT_COUNT; segment++)
            if(state->lastSegmentScore[segment] > maximum)
                maximum = state->lastSegmentScore[segment];
        twoLargestSegments(state, &primary, &secondary);
        compact = compactUniqueResponse(state, primary, secondary);

        if(maximum >= state->activateThresholdCodes)
        {
            state->activityLatched = 1U;
            state->activePrimarySegment = primary;
            state->activeSecondarySegment = secondary;
            state->lowActivityFrames = 0U;

            if(!wasLatched)
            {
                /* A new contact always receives the two-triplet TRACK13
                 * profile first.  TRACK11 is earned only from eight later,
                 * complete and mutually consistent wide frames. */
                state->singleTrackActive = 0U;
                state->compactUniqueFrames = 0U;
                state->compactCandidateSegment =
                        STRESS_MULTIRATE_NO_SEGMENT;
            }
            else if(state->currentPlan.profile ==
                    STRESS_MULTIRATE_PROFILE_TRACK_SINGLE)
            {
                /* Any second component, changed winner, high effective area,
                 * or other ambiguity restores full two-ROI tracking on the
                 * very next frame. */
                if(!state->singleTrackEnabled || !compact ||
                   primary != state->currentPlan.primarySegment)
                {
                    state->singleTrackActive = 0U;
                    state->compactUniqueFrames = 0U;
                    state->compactCandidateSegment =
                            STRESS_MULTIRATE_NO_SEGMENT;
                }
            }
            else if(state->currentPlan.profile ==
                    STRESS_MULTIRATE_PROFILE_TRACK_WIDE &&
                    state->singleTrackEnabled && compact)
            {
                if(state->compactCandidateSegment == primary)
                {
                    if(state->compactUniqueFrames < 0xFFU)
                        state->compactUniqueFrames++;
                }
                else
                {
                    state->compactCandidateSegment = primary;
                    state->compactUniqueFrames = 1U;
                }
                if(state->compactUniqueFrames >=
                   STRESS_MULTIRATE_SINGLE_CONFIRM_FRAMES)
                    state->singleTrackActive = 1U;
            }
            else
            {
                state->singleTrackActive = 0U;
                state->compactUniqueFrames = 0U;
                state->compactCandidateSegment =
                        STRESS_MULTIRATE_NO_SEGMENT;
            }
        }
        else if(state->activityLatched &&
                maximum <= state->releaseThresholdCodes)
        {
            /* Silence is hysteresis, not spatial ambiguity.  Keep the current
             * sparse profile for exactly the four-frame release qualification
             * and then return to SURVEY9. */
            if(state->lowActivityFrames < 0xFFU)
                state->lowActivityFrames++;
            state->compactUniqueFrames = 0U;
            state->compactCandidateSegment = STRESS_MULTIRATE_NO_SEGMENT;
            if(state->lowActivityFrames >= state->releaseFrames)
            {
                memset(state->lastSegmentScore, 0,
                       sizeof(state->lastSegmentScore));
                state->activityLatched = 0U;
                state->activePrimarySegment = STRESS_MULTIRATE_NO_SEGMENT;
                state->activeSecondarySegment = STRESS_MULTIRATE_NO_SEGMENT;
                state->singleTrackActive = 0U;
            }
        }
        else if(state->activityLatched)
        {
            /* Inside the activation/release hysteresis band, a SINGLE frame
             * may stay single only while the same unique spatial component
             * remains unambiguous.  Weak frames cannot earn TRACK11. */
            state->lowActivityFrames = 0U;
            state->compactUniqueFrames = 0U;
            state->compactCandidateSegment = STRESS_MULTIRATE_NO_SEGMENT;
            if(state->currentPlan.profile ==
               STRESS_MULTIRATE_PROFILE_TRACK_SINGLE &&
               (!state->singleTrackEnabled || !compact ||
                primary != state->currentPlan.primarySegment))
                state->singleTrackActive = 0U;
        }
    }
    else
    {
        /* A missing sentinel or an ADC value outside the 12-bit domain can
         * never prove silence or uniqueness.  Preserve the contact latch,
         * cancel TRACK11 immediately, and forbid baseline learning. */
        maximum = (uint16_t)(state->releaseThresholdCodes + 1U);
        state->baselineQuietFrames = 0U;
        state->lowActivityFrames = 0U;
        state->singleTrackActive = 0U;
        state->compactUniqueFrames = 0U;
        state->compactCandidateSegment = STRESS_MULTIRATE_NO_SEGMENT;
    }

    /* Track very slow optical/thermal drift only after several consecutive
     * no-contact frames.  Update fresh rows only: cached samples must never
     * leak into the current baseline.  Q6 state avoids integer dead-band while
     * implementing alpha=1/64. */
    if(sentinelsCompleteAndInDomain(state) && !state->activityLatched &&
       maximum <= state->releaseThresholdCodes)
    {
        if(state->baselineQuietFrames < 0xFFU)
            state->baselineQuietFrames++;
        if(state->baselineQuietFrames >=
           STRESS_MULTIRATE_BASELINE_QUIET_FRAMES)
        {
            /* A forced-ready startup becomes fully qualified once the normal
             * quiet detector has subsequently observed four stable frames. */
            if(state->baselineStatus ==
               STRESS_MULTIRATE_BASELINE_FORCED_READY)
                state->baselineStatus = STRESS_MULTIRATE_BASELINE_READY;
            for(uint16_t point = 0U; point < STRESS_TABLE_POINT_COUNT; point++)
            {
                int32_t targetQ6;
                int32_t baselineQ6;
                int32_t deltaQ6;
                if(!bitIsSet(state->currentPlan.freshBitmap, point)) continue;
                targetQ6 = (int32_t)state->cachedCh1[point] <<
                        STRESS_MULTIRATE_BASELINE_EMA_SHIFT;
                baselineQ6 =
                        (int32_t)state->baselineAccumulatorQ6[point];
                deltaQ6 = targetQ6 - baselineQ6;
                baselineQ6 += deltaQ6 /
                        (1L << STRESS_MULTIRATE_BASELINE_EMA_SHIFT);
                state->baselineAccumulatorQ6[point] = (uint32_t)baselineQ6;
                state->baselineCh1[point] = (uint16_t)(
                        (state->baselineAccumulatorQ6[point] +
                         (1UL << (STRESS_MULTIRATE_BASELINE_EMA_SHIFT - 1U))) >>
                        STRESS_MULTIRATE_BASELINE_EMA_SHIFT);
            }
        }
    }
    else
    {
        state->baselineQuietFrames = 0U;
    }
}

uint8_t StressMultirate_IsEnabled(const StressMultirateState_t *state)
{
    return (state != 0 && state->enabled) ? 1U : 0U;
}

const StressMultiratePlan_t *StressMultirate_CurrentPlan(
        const StressMultirateState_t *state)
{
    return state ? &state->currentPlan : 0;
}

StressMultirateBaselineStatus_t StressMultirate_BaselineStatus(
        const StressMultirateState_t *state)
{
    if(state == 0) return STRESS_MULTIRATE_BASELINE_LEARNING;
    return (StressMultirateBaselineStatus_t)state->baselineStatus;
}
