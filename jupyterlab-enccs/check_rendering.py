#!/usr/bin/env python3
"""Check the lesson's MyST rendering in a private JupyterLab session."""

import argparse
import os
from pathlib import Path
from urllib.parse import quote

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Private JupyterLab base URL")
    token_source = parser.add_mutually_exclusive_group(required=True)
    token_source.add_argument("--token-file", type=Path)
    token_source.add_argument("--token-env", help="name of a private token environment variable")
    parser.add_argument("--geckodriver", default="geckodriver")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    token = (
        args.token_file.read_text(encoding="utf-8").strip()
        if args.token_file is not None
        else os.environ.get(args.token_env, "").strip()
    )
    if not token:
        raise SystemExit("empty Jupyter token")
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
                + f"/lab/workspaces/mlip-render-{os.getpid()}/tree/"
                + quote(relative, safe="/")
                + "?token="
                + quote(token, safe="")
            )
            driver.get(url)
            try:
                WebDriverWait(driver, 30).until(
                    lambda current: any(
                        heading.text == expected_heading
                        for heading in current.find_elements(
                            "css selector", ".jp-Notebook .myst h1"
                        )
                    )
                )
            except TimeoutException as error:
                observed = driver.execute_script(
                    """return {
                        notebooks: document.querySelectorAll('.jp-Notebook').length,
                        markdownViewers: document.querySelectorAll('.jp-MarkdownViewer').length,
                        headings: [...document.querySelectorAll('h1')]
                            .map(item => item.textContent.trim()).slice(0, 6)
                    };"""
                )
                raise SystemExit(f"notebook did not open: {relative}: {observed}") from error
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
            required_callout = {
                "setup/index.md": "No notebook submits a Slurm job",
                "episodes/08-scaling.md": "Do not treat",
            }.get(relative)
            if required_callout and not driver.execute_script(
                """
                const heading = [...document.querySelectorAll('.jp-Notebook .myst h1')]
                    .find(item => item.textContent.trim() === arguments[0]);
                const root = heading.closest('.jp-Notebook');
                return [...root.querySelectorAll('aside')]
                    .some(item => item.textContent.includes(arguments[1]));
                """,
                expected_heading,
                required_callout,
            ):
                raise SystemExit(f"required callout missing: {relative}")
            if relative == "episodes/02-alchemi-image.md" and not driver.execute_script(
                """
                const heading = [...document.querySelectorAll('.jp-Notebook .myst h1')]
                    .find(item => item.textContent.trim() === arguments[0]);
                const root = heading.closest('.jp-Notebook');
                return [...root.querySelectorAll('pre')]
                    .some(item => item.textContent.includes('13 | staged=') &&
                        item.textContent.includes('17 | apptainer build'));
                """,
                expected_heading,
            ):
                raise SystemExit("original source line numbers missing in notebook")
            if page.name == "renderer-fixture.md" and (
                state["callouts"] != 3 or not state["source"]
            ):
                raise SystemExit(f"ENCCS fixture incomplete: {state}")
            if relative == "setup/index.md":
                count = driver.execute_script(
                    """
                    const heading = [...document.querySelectorAll('.jp-Notebook .myst h1')]
                        .find(item => item.textContent.trim() === 'Before you start');
                    const root = heading.closest('.jp-Notebook');
                    const rows = root.querySelectorAll('.myst-tab-set-row');
                    if (rows.length !== 2) return rows.length;
                    [...rows[0].querySelectorAll('.myst-tab-item-header')]
                        .find(item => item.textContent.trim() === 'JUPITER').click();
                    return rows.length;
                    """
                )
                if count != 2:
                    raise SystemExit(f"expected two site tab sets; found {count}")
                WebDriverWait(driver, 10).until(
                    lambda current: current.execute_script(
                        """
                        const heading = [...document.querySelectorAll('.jp-Notebook .myst h1')]
                            .find(item => item.textContent.trim() === 'Before you start');
                        const root = heading.closest('.jp-Notebook');
                        return [...root.querySelectorAll('.myst-tab-set-row')]
                            .every(row => [...row.querySelectorAll('.myst-tab-item-header')]
                                .some(item => item.textContent.trim() === 'JUPITER' &&
                                    item.className.includes('header-active')));
                        """
                    )
                )
            print(f"PASS {relative}")
        print(f"PASS: {len(pages)} pages, themed heading, included code with source lines, callouts, synchronized site tabs")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
