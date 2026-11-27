# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Check the interpolation RMS recorded in elevation history."""

import re
from collections import Counter
from math import cos, fsum, sin, sqrt
from subprocess import PIPE

import pytest

import grass.script as gs


def test_parallel_rms_history(xy_dataset_session):
    """RMS covers all fitting residuals for disjoint, populated segments."""
    env = xy_dataset_session.env.copy()
    env.update(
        LC_ALL="C",
        LANGUAGE="C",
        GRASS_MESSAGE_FORMAT="plain",
        OMP_DYNAMIC="FALSE",
        OMP_THREAD_LIMIT="4",
    )
    gs.run_command("g.region", n=320, s=0, e=320, w=0, res=2, env=env)
    points = {
        (x, y): (
            75
            + 0.001 * (x - 160) ** 2
            - 0.002 * (y - 160) ** 2
            + 30 * sin(x / 17) * cos(y / 23)
        )
        for y in range(5, 320, 10)
        for x in range(5, 320, 10)
    }
    gs.write_command(
        "v.in.ascii",
        flags="zt",
        input="-",
        output="rms_points",
        separator="pipe",
        z=3,
        stdin="".join(f"{x}|{y}|{z:.17g}\n" for (x, y), z in points.items()),
        env=env,
    )
    process = gs.start_command(
        "v.surf.rst",
        input="rms_points",
        elevation="rms_elevation",
        deviations="rms_deviations",
        overwin="rms_windows",
        nprocs=4,
        segmax=16,
        npmin=17,
        smooth=0.5,
        dmin=0.1,
        verbose=True,
        stderr=PIPE,
        env=env,
    )
    _, raw_messages = process.communicate()
    messages = gs.decode(raw_messages)
    assert process.returncode == 0, messages
    if "GRASS is not compiled with OpenMP support" in messages:
        pytest.skip("v.surf.rst requires OpenMP for this regression")
    assert "Processing segments in parallel" in messages
    assert "Number of points being used 1024" in messages, messages

    windows = gs.read_command(
        "v.out.ascii",
        input="rms_windows",
        layer=-1,
        format="standard",
        env=env,
    )
    # Each exported rectangle describes the actual fitting window of a leaf.
    lines = windows.splitlines()
    features = iter(lines[lines.index("VERTI:") + 1 :])
    memberships = Counter()
    counts = []
    for line in features:
        assert line.split() == ["L", "5"]
        vertices = [tuple(map(float, next(features).split())) for _ in range(5)]
        xs, ys = zip(*vertices, strict=True)
        west, east = min(xs), max(xs)
        south, north = min(ys), max(ys)
        selected = [(x, y) for x, y in points if west < x < east and south < y < north]
        memberships.update(selected)
        counts.append(len(selected))
    assert counts == [16] * 64
    assert memberships == Counter(points.keys())

    exported = gs.read_command(
        "v.out.ascii",
        input="rms_deviations",
        columns="flt1",
        separator="pipe",
        precision=17,
        env=env,
    )
    residuals = {}
    for line in exported.splitlines():
        x, y, _, _, residual = line.split("|")
        xy = (round(float(x)), round(float(y)))
        assert float(x) == pytest.approx(xy[0], rel=0, abs=1e-8)
        assert float(y) == pytest.approx(xy[1], rel=0, abs=1e-8)
        assert xy not in residuals
        residuals[xy] = abs(float(residual))
    assert residuals.keys() == points.keys()
    assert max(residuals.values()) > 0.1

    history = gs.read_command("r.info", flags="h", map="rms_elevation", env=env)
    match = re.search(r"rmsdevi=(\d+\.\d+)", history)
    assert match, history
    rms = float(match[1])
    # Both SQL residuals and raster history use six decimal places.
    half_unit = 0.5e-6
    lower = sqrt(
        fsum(max(value - half_unit, 0) ** 2 for value in residuals.values())
        / len(points)
    )
    upper = sqrt(
        fsum((value + half_unit) ** 2 for value in residuals.values()) / len(points)
    )
    rounding = half_unit + 1e-12
    assert lower - rounding <= rms <= upper + rounding, (
        f"History RMS {rms:.6f} is outside the expected interval "
        f"[{lower - rounding:.12f}, {upper + rounding:.12f}]"
    )
