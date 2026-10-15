"""Research-only career features from verified identities and earlier results."""
from bisect import bisect_left
from collections import Counter, defaultdict
import math

import pandas as pd

from external_mma.identity import normalize_name
from external_mma.integration import is_ufc_promotion, load_identity_map
from external_mma.storage import ExternalMmaStore
from .point_in_time import PointInTimeDatasetBuilder, _identity_token


FEATURES = ('external_history_present', 'external_fights_log',
            'external_win_balance', 'external_opponent_rating', 'career_rating')


def load_career_bouts(raw, roots):
    """Keep source identities separate unless an approved crosswalk connects them.

    Date + unordered names identifies duplicate observations, never a career ID.
    UFC witnesses are excluded even when a source calls them Road to UFC.
    """
    bouts, seen = [], {}
    counts = Counter()
    promotions = Counter()
    linked_by_promotion = Counter()
    linked_fighters = set()
    def key(date, a, b):
        return (str(date)[:10], tuple(sorted((normalize_name(a), normalize_name(b)))))
    for _, rows in raw.groupby('fight_url', sort=False):
        if len(rows) != 2:
            raise ValueError('Expected mirrored UFC rows')
        r = rows.iloc[0]
        k = key(r['date'], r['fighter'], r['opponent'])
        seen[k] = 'ufc'
        bouts.append((k[0], _identity_token(r.fighter_url),
                      _identity_token(r.opponent_url), r.result, False))
    for root in roots:
        mapping = load_identity_map(root/'identity_map.csv')
        counts['approved_identity_links'] += len(mapping)
        for r in ExternalMmaStore(root).observations():
            if r.discipline != 'mma' or not r.professional:
                raise ValueError('Career history requires professional MMA results')
            if is_ufc_promotion(r.promotion):
                counts['ufc_witnesses_excluded'] += 1
                continue
            k = key(r.event_date, r.fighter_name, r.opponent_name)
            # Normalize the result to alphabetical name order for comparisons.
            score = {'W': 1, 'L': 0, 'D': .5, 'NC': None}[r.result]
            signature = score if normalize_name(r.fighter_name) == k[1][0] or score is None else 1-score
            if k in seen:
                if seen[k] != 'ufc' and seen[k] != signature:
                    raise ValueError(f'Conflicting external results for {k}')
                counts['duplicate_bouts_excluded'] += 1
                continue
            seen[k] = signature
            ids = []
            for source_id in (r.fighter_source_id, r.opponent_source_id):
                canonical = mapping.get((r.source, source_id))
                counts['linked_participants' if canonical else 'unlinked_participants'] += 1
                if canonical:
                    linked_by_promotion[r.promotion] += 1
                    linked_fighters.add(canonical)
                ids.append(canonical or f'external:{r.source}:{source_id}')
            if ids[0] == ids[1]:
                raise ValueError('Identity map collapsed opponents')
            bouts.append((r.event_date, *ids, r.result, True))
            promotions[r.promotion] += 1
    return bouts, {**dict(counts), 'external_bouts': sum(promotions.values()),
                   'linked_fighters': len(linked_fighters),
                   'linked_participants_by_promotion': dict(sorted(linked_by_promotion.items())),
                   'promotions': dict(sorted(promotions.items()))}


class CareerHistory:
    """All features for date D exclude every result on D and later dates."""
    def __init__(self, bouts):
        self.dates, self.values = defaultdict(list), defaultdict(list)
        ratings = defaultdict(lambda: 1500.)
        records = defaultdict(lambda: [0, 0., 0.])
        by_date = defaultdict(list)
        for bout in bouts:
            by_date[bout[0]].append(bout)
        for date, group in sorted(by_date.items()):
            changes = defaultdict(float)
            touched = set()
            for _, a, b, result, external in sorted(group):
                score = {'W': 1., 'L': 0., 'D': .5, 'NC': None}[result]
                if score is None:
                    continue
                ra, rb = ratings[a], ratings[b]
                change = 32 * (score - 1/(1+10**((rb-ra)/400)))
                changes[a] += change
                changes[b] -= change
                touched.update((a,b))
                if external:
                    for fighter, s, opponent_rating in ((a,score,rb),(b,1-score,ra)):
                        r = records[fighter]
                        r[0] += 1
                        r[1] += s
                        r[2] += opponent_rating-1500
            for fighter in sorted(touched):
                ratings[fighter] += changes[fighter]
                n, wins, strength = records[fighter]
                self.dates[fighter].append(date)
                self.values[fighter].append((float(n>0), math.log1p(n),
                    (wins-.5*n)/(n+4), strength/(n+4)/400,
                    (ratings[fighter]-1500)/400))

    def features(self, fighter, date):
        index = bisect_left(self.dates.get(fighter, []), str(date)[:10])-1
        return self.values[fighter][index] if index >= 0 else (0.,)*len(FEATURES)


class CareerDatasetBuilder(PointInTimeDatasetBuilder):
    def __init__(self, raw_fights, fighter_stats, history):
        self.career_history = history
        super().__init__(raw_fights, fighter_stats)

    def _side_features(self, fighter_id, fight_date, division):
        values = super()._side_features(fighter_id, fight_date, division)
        values.update(zip(FEATURES, self.career_history.features(fighter_id, fight_date)))
        return values

    def extend_baseline(self, frame):
        """Append research inputs without changing any original feature or label."""
        output = frame.copy()
        rows = []
        coverage = []
        for r in frame.itertuples():
            a = self.career_history.features(r.fighter_id, r.date)
            b = self.career_history.features(r.opponent_id, r.date)
            rows.append([x-y for x,y in zip(a,b)])
            coverage.append(bool(a[0] or b[0]))
        output[[name+'_diff' for name in FEATURES]] = pd.DataFrame(rows, index=output.index)
        output['external_history_available'] = coverage
        return output
