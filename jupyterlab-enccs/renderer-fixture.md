---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Renderer fixture

:::{objectives}
The objective callout stays visible in the notebook.
:::

:::{questions}
Does the included source appear below?
:::

:::{keypoints}
The published page and notebook read the same Markdown source.
:::

```{literalinclude} ../examples/alchemi_si_one_cell.py
:language: python
:lines: 1-5
:lineno-match:
```

::::{tab-set}
:sync-group: site
:::{tab-item} Arrhenius
:sync: arrhenius
First site.
:::
:::{tab-item} JUPITER
:sync: jupiter
Second site.
:::
::::

::::{tab-set}
:sync-group: site
:::{tab-item} Arrhenius
:sync: arrhenius
First route.
:::
:::{tab-item} JUPITER
:sync: jupiter
Second route.
:::
::::
