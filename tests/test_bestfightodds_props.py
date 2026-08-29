from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bestfightodds_props import (  # noqa: E402
    METHODS,
    PropParseError,
    complete_method_markets,
    method_from_label,
    parse_american_moneyline,
    parse_bestfightodds_method_props,
)


def _fixture(*, conflicting_duplicate: bool = False) -> str:
    method_rows = []
    prop_type = 8
    labels = {
        (1, "ko_tko"): "Alpha wins by TKO/KO",
        (1, "submission"): "Alpha wins by submission",
        (1, "decision"): "Alpha wins by decision",
        (2, "ko_tko"): "Beta wins by KO/TKO/DQ",
        (2, "submission"): "Beta wins by submission",
        (2, "decision"): "Beta wins by decision",
    }
    for index, ((side, _), label) in enumerate(labels.items()):
        line = 200 + index * 50
        second = line + (1 if conflicting_duplicate and index == 0 else 0)
        method_rows.append(
            f"""
            <tr class="pr"><th scope="row">{label}</th>
              <td class="but-sgp" data-li="[21,{side},900,{prop_type},1]">+{line}▲</td>
              <td class="but-sgp" data-li="[22,{side},900,{prop_type},1]">+{line + 10}</td>
              <td class="button-cell but-sip" data-li="[{side},900,{prop_type},1]"></td>
            </tr>
            <tr class="pr"><th scope="row">Any other result</th>
              <td class="but-sgp" data-li="[21,{side},900,{prop_type},2]">-110</td>
            </tr>
            """
        )
        if conflicting_duplicate and index == 0:
            method_rows.append(
                f"""
                <tr class="pr"><th scope="row">{label}</th>
                  <td class="but-sgp" data-li="[21,{side},900,{prop_type},1]">+{second}</td>
                </tr>
                """
            )
        prop_type += 1
    # The first table models the label-only mobile copy seen on the live page.
    return f"""
    <html><body>
      <table><tr class="pr"><th scope="row">Alpha wins by TKO/KO</th></tr></table>
      <table class="odds-table">
        <tr><th></th><th data-b="21"><span>Book A</span></th>
            <th data-b="22"><span>Book B</span></th></tr>
        <tr><th scope="row"><a href="/fighters/alpha">Fighter Alpha</a></th>
          <td class="but-sg" data-li="[21,1,900]">-125</td></tr>
        <tr><th scope="row"><a href="/fighters/beta">Fighter Beta</a></th>
          <td class="but-sg" data-li="[21,2,900]">+105</td></tr>
        {''.join(method_rows)}
        <tr class="pr"><th scope="row">Alpha wins by TKO/KO in round 1</th>
          <td class="but-sgp" data-li="[21,1,900,99,1]">+500</td></tr>
        <tr class="pr"><th scope="row">Alpha wins by unanimous decision</th>
          <td class="but-sgp" data-li="[21,1,900,100,1]">+600</td></tr>
      </table>
    </body></html>
    """


class BestFightOddsPropTests(unittest.TestCase):
    def test_moneyline_parser_handles_arrows_even_and_invalid_values(self):
        self.assertEqual(parse_american_moneyline("+250 ▼"), 250)
        self.assertEqual(parse_american_moneyline("−135▲"), -135)
        self.assertEqual(parse_american_moneyline("EVEN"), 100)
        self.assertIsNone(parse_american_moneyline("n/a"))
        self.assertIsNone(parse_american_moneyline("1.91"))

    def test_method_classifier_excludes_round_and_decision_subtypes(self):
        self.assertEqual(method_from_label("A wins by TKO/KO"), "ko_tko")
        self.assertEqual(method_from_label("A wins by KO/TKO/DQ"), "ko_tko")
        self.assertEqual(method_from_label("A wins by submission"), "submission")
        self.assertEqual(method_from_label("A wins by decision"), "decision")
        self.assertIsNone(method_from_label("A wins by TKO/KO in round 1"))
        self.assertIsNone(method_from_label("A wins by unanimous decision"))

    def test_parser_builds_only_six_primary_method_selections(self):
        parsed = parse_bestfightodds_method_props(_fixture())
        self.assertEqual(len(parsed), 6)
        self.assertEqual({item.method for item in parsed}, set(METHODS))
        self.assertEqual({item.source_fighter_side for item in parsed}, {1, 2})
        self.assertTrue(all(item.mean_history_available for item in parsed))
        self.assertTrue(all(len(item.book_prices) == 2 for item in parsed))
        self.assertEqual(parsed[0].fighter_1_name, "Fighter Alpha")
        self.assertEqual(parsed[0].fighter_2_name, "Fighter Beta")

        complete = complete_method_markets(parsed)
        self.assertEqual(set(complete), {(900, 21), (900, 22)})
        self.assertEqual(len(complete[(900, 21)]), 6)

    def test_parser_rejects_conflicting_duplicate_price(self):
        with self.assertRaises(PropParseError):
            parse_bestfightodds_method_props(_fixture(conflicting_duplicate=True))


if __name__ == "__main__":
    unittest.main()
