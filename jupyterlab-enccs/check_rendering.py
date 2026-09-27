#!/usr/bin/env python3
"""Check the lesson's MyST rendering in a private JupyterLab session."""

import argparse
from pathlib import Path
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Private JupyterLab base URL")
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--geckodriver", default="geckodriver")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("empty token file")
    pages = [
        page
        for folder in ("episodes", "setup", "reference")
        for page in sorted((root / folder).glob("*.md"))
    ]
    pages.append(root / "jupyterlab-enccs" / "renderer-fixture.md")

    options = Options()
    options.add_argument("-headless")
    driver = webdriver.Firefox(options=options, service=Service(args.geckodriver))
    try:
        driver.set_window_size(1440, 1100)
        for page in pages:
            relative = page.relative_to(root).as_posix()
            expected_heading = next(
                line[2:].strip()
                for line in page.read_text(encoding="utf-8").splitlines()
                if line.startswith("# ")
            )
            url = (
                args.base_url.rstrip("/")
                + "/lab/tree/"
                + quote(relative, safe="/")
                + "?token="
                + quote(token, safe="")
            )
            driver.get(url)
            WebDriverWait(driver, 30).until(
                lambda current: any(
                    heading.text == expected_heading
                    for heading in current.find_elements(
                        "css selector", ".jp-Notebook .myst h1"
                    )
                )
            )
            if page.name == "renderer-fixture.md":
                WebDriverWait(driver, 15).until(
                    lambda current: current.execute_script(
                        """
                        const heading = [...document.querySelectorAll('.jp-Notebook .myst h1')]
                            .find(item => item.textContent.trim() === arguments[0]);
                        return heading?.closest('.jp-Notebook')?.querySelectorAll(
                            'aside.enccs-questions, aside.enccs-objectives, aside.enccs-keypoints'
                        ).length === 3;
                        """,
                        expected_heading,
                    )
                )
            state = driver.execute_script(
                """
                const heading = [...document.querySelectorAll('.jp-Notebook .myst h1')]
                    .find(item => item.textContent.trim() === arguments[0]);
                const root = heading.closest('.jp-Notebook');
                return {
                    border: getComputedStyle(heading).borderBottomWidth,
                    missing: root.innerText.includes('literalinclude unavailable'),
                    source: root.innerText.includes('import argparse'),
                    callouts: root.querySelectorAll(
                        'aside.enccs-questions, aside.enccs-objectives, aside.enccs-keypoints'
                    ).length
                };
                """,
                expected_heading,
            )
            if state["border"] != "3px" or state["missing"]:
                raise SystemExit(f"rendering failed: {relative}: {state}")
            if page.name == "renderer-fixture.md" and (
                state["callouts"] != 3 or not state["source"]
            ):
                raise SystemExit(f"ENCCS fixture incomplete: {state}")
            print(f"PASS {relative}")
        print(f"PASS: {len(pages)} pages, themed heading, included code, three callouts")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
