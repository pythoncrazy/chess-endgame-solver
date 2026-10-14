"""Browser smoke test of the visualizer: load, click-move, drag, editor, buttons.

Usage: python -I tools/visualizer/ui_smoke.py URL
Prints every console error and failed request; exits non-zero on any failure.
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import Page, sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8642/"
problems: list[str] = []


def square(page: Page, name: str):
    index = "abcdefgh".index(name[0]) + 8 * (int(name[1]) - 1)
    return page.locator(f'.sq[data-sq="{index}"]')


def text(page: Page, selector: str) -> str:
    return page.locator(selector).inner_text().strip()


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"{'ok  ' if condition else 'FAIL'} {label} {detail}")
    if not condition:
        problems.append(label)


def main() -> int:  # noqa: PLR0915 - one linear browser script
    with sync_playwright() as p:
        engine = getattr(p, os.environ.get("BROWSER", "chromium"))
        browser = engine.launch(executable_path=os.environ.get("CHROME") or None)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}"))
        page.goto(URL + "#8%2F8%2F8%2F4k3%2F8%2F8%2F8%2FKQ6%20w%20-%20-%200%201")
        page.wait_for_selector(".tb-move", timeout=10000)
        check("loads with moves", page.locator(".tb-move").count() > 0)
        check(
            "pearl shows a result", text(page, "#pearl") not in ("", "\u2013"), text(page, "#pearl")
        )
        # Click-to-move: queen b1 -> b5.
        square(page, "b1").click()
        check("selecting shows destinations", page.locator(".sq.dest").count() > 0)
        square(page, "b5").click()
        page.wait_for_timeout(800)
        check("click move played", "Q" in text(page, "#tview"), text(page, "#tview"))
        check(
            "turn flipped to black", page.locator("#turn-b").get_attribute("aria-pressed") == "true"
        )
        # Explorer row click plays the move.
        before = page.locator("#tview button").count()
        page.locator(".tb-move").first.click()
        page.wait_for_timeout(800)
        check("explorer click plays move", page.locator("#tview button").count() == before + 1)
        # The played move's value comes from the parent reply, before the child's.
        page.route("**/api/probe*", lambda route: (page.wait_for_timeout(1500), route.continue_()))
        page.locator(".tb-move").first.click()
        page.wait_for_timeout(300)
        check(
            "value shown while the move list loads",
            text(page, "#pearl") not in ("", "\u2013") and "Loading" in text(page, "#ceval-sub"),
            f"{text(page, '#pearl')} / {text(page, '#ceval-sub')}",
        )
        page.wait_for_timeout(2000)
        page.unroute("**/api/probe*")
        check("move list arrives", page.locator(".tb-move").count() > 0)
        # Navigation.
        page.click("#prev")
        page.wait_for_timeout(500)
        check("prev moves back", page.locator("#tview button.active").count() == 1)
        page.click("#first")
        page.wait_for_timeout(500)
        # Drag: queen b1 -> h7.
        src, dst = square(page, "b1").bounding_box(), square(page, "h7").bounding_box()
        page.mouse.move(src["x"] + 20, src["y"] + 20)
        page.mouse.down()
        page.mouse.move(dst["x"] + 20, dst["y"] + 20, steps=8)
        page.mouse.up()
        page.wait_for_timeout(800)
        check("drag move played", square(page, "h7").locator("svg").count() == 1)
        # Best line.
        page.goto(URL + "#8%2F8%2F8%2F4k3%2F8%2F8%2F8%2FKQ6%20w%20-%20-%200%201")
        page.wait_for_selector(".tb-move")
        page.click("#best")
        page.wait_for_timeout(2500)
        check("best line ends in mate", "#" in text(page, "#tview"), text(page, "#tview")[-60:])
        # Editor: clear, place pieces, side to move.
        page.click("#clear")
        page.wait_for_timeout(400)
        page.click('[data-tool="wK"]')
        square(page, "e1").click()
        page.click('[data-tool="wR"]')
        square(page, "a1").click()
        page.click('[data-tool="bK"]')
        square(page, "e8").click()
        page.wait_for_timeout(800)
        check(
            "editor builds KRvK",
            text(page, "#side-material") in ("KvKR", "KRvK"),
            text(page, "#side-material"),
        )
        check(
            "FEN updated",
            page.input_value("#fen").startswith("4k3/8/8/8/8/8/8/R3K3"),
            page.input_value("#fen"),
        )
        # Random position.
        page.select_option("#material", "KQBvKR") if page.locator(
            '#material option[value="KQBvKR"]'
        ).count() else None
        page.click("#random-form button")
        page.wait_for_timeout(1500)
        check("random position loads", page.locator(".tb-move").count() > 0)
        # Other views.
        page.click('[data-view="materials"]')
        page.wait_for_timeout(500)
        check("materials table", page.locator("#materials-table tbody tr").count() > 0)
        page.click('[data-view="evidence"]')
        page.wait_for_timeout(1500)
        check("evidence chart", page.locator("#chart-changed svg").count() == 1)
        browser.close()
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
