# pylint: disable=invalid-name,too-many-statements
import builtins
import os
import sys
import types
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from rocketpy.plots.compare import Compare
from rocketpy.plots.plot_helpers import (
    show_or_save_animation,
    show_or_save_fig,
    show_or_save_plot,
)


@patch("matplotlib.pyplot.show")
def test_compare(mock_show, flight_calisto):  # pylint: disable=unused-argument
    """Here we want to test the 'x_attributes' argument, which is the only one
    that is not tested in the other tests.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    flight_calisto : rocketpy.Flight
        Flight object to be used in the tests. See conftest.py for more details.
    """
    flight = flight_calisto

    objects = [flight, flight, flight]

    comparison = Compare(object_list=objects)

    fig, _ = comparison.create_comparison_figure(
        y_attributes=["z"],
        n_rows=1,
        n_cols=1,
        figsize=(10, 10),
        legend=False,
        title="Test",
        x_labels=["Time (s)"],
        y_labels=["Altitude (m)"],
        x_lim=(0, 3),
        y_lim=(0, 1000),
        x_attributes=["time"],
    )

    assert isinstance(fig, plt.Figure)


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("filename", [None, "test.png"])
def test_show_or_save_plot(mock_show, filename):
    """This test is to check if the show_or_save_plot function is
    working properly.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing
        the plots.
    filename : str
        Name of the file to save the plot. If None, the plot will be
        shown instead.
    """
    plt.subplots()
    show_or_save_plot(filename)

    if filename is None:
        mock_show.assert_called_once()
    else:
        assert os.path.exists(filename)
        os.remove(filename)


@pytest.mark.parametrize("filename", [None, "test.png"])
def test_show_or_save_fig(filename):
    """This test is to check if the show_or_save_fig function is
    working properly.

    Parameters
    ----------
    filename : str
        Name of the file to save the plot. If None, the plot will be
        shown instead.
    """
    fig, _ = plt.subplots()

    fig.show = MagicMock()
    show_or_save_fig(fig, filename)

    if filename is None:
        fig.show.assert_called_once()
    else:
        assert os.path.exists(filename)
        os.remove(filename)


@pytest.mark.parametrize("filename", [None, "test.gif"])
@patch("matplotlib.pyplot.show")
def test_show_or_save_animation(mock_show, filename):
    """This test is to check if the show_or_save_animation function is
    working properly.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the animation.
    filename : str
        Name of the file to save the animation. If None, the animation will be
        shown instead.
    """

    # Create a simple animation object
    fig, ax = plt.subplots()

    def update(frame):
        ax.plot([0, frame], [0, frame])
        return ax

    animation = FuncAnimation(fig, update, frames=5)

    show_or_save_animation(animation, filename)

    if filename is None:
        mock_show.assert_called_once()
    else:
        assert os.path.exists(filename)
        os.remove(filename)


def test_show_or_save_animation_unsupported_format():
    # Test that show_or_save_animation raises ValueError for unsupported formats.
    fig, ax = plt.subplots()

    def update(frame):
        ax.plot([0, frame], [0, frame])
        return ax

    animation = FuncAnimation(fig, update, frames=5)

    with pytest.raises(ValueError, match="Unsupported file ending"):
        show_or_save_animation(animation, "test.mp4")


def test_animate_propellant_mass(cesaroni_m1670, monkeypatch):
    """Test that animate_propellant_mass saves a .gif file correctly."""

    def mock_show_or_save(animation, filename=None, fps=30):  # pylint: disable=unused-argument
        if filename:
            with open(filename, "a"):
                pass

    monkeypatch.setattr(
        "rocketpy.plots.motor_plots.show_or_save_animation", mock_show_or_save
    )

    motor = cesaroni_m1670
    animation = motor.plots.animate_propellant_mass(filename="cesaroni_m1670.gif")

    # Check animation type
    assert isinstance(animation, FuncAnimation)

    # check if file exists
    assert os.path.exists("cesaroni_m1670.gif")

    os.remove("cesaroni_m1670.gif")


def test_animate_fluid_volume(example_mass_flow_rate_based_tank_seblm, monkeypatch):
    """Test that animate_fluid_volume saves a .gif file correctly."""

    def mock_show_or_save(animation, filename=None, fps=30):  # pylint: disable=unused-argument
        if filename:
            with open(filename, "a"):
                pass

    monkeypatch.setattr(
        "rocketpy.plots.tank_plots.show_or_save_animation", mock_show_or_save
    )

    tank = example_mass_flow_rate_based_tank_seblm
    animation = tank.plots.animate_fluid_volume(filename="test_fluid_volume.gif")

    # Check animation type
    assert isinstance(animation, FuncAnimation)

    # Check if file exists
    assert os.path.exists("test_fluid_volume.gif")

    os.remove("test_fluid_volume.gif")


class _DummyPyVistaMesh:
    """Minimal mutable mesh used by the PyVista animation tests."""

    def __init__(self, *_args, **_kwargs):
        self.center = (0.0, 0.0, 2.0)
        self.length = 4.0
        self.points = np.asarray(_args[0]) if _args else np.empty((0, 3))
        self.lines = _kwargs.get("lines")
        self.copy_count = 0
        self.visible = True
        self.texture_mapped = False
        self.point_data = {}

    def translate(self, *_args, **_kwargs):
        return self

    def scale(self, factor, *_args, **_kwargs):
        self.length *= factor
        return self

    def transform(self, *_args, **_kwargs):
        transformed = _DummyPyVistaMesh()
        transformed.length = self.length
        return transformed

    def copy_from(self, *_args, **_kwargs):
        self.copy_count += 1

    def texture_map_to_plane(self, *_args, **_kwargs):
        self.texture_mapped = True
        return self

    def SetVisibility(self, visible):
        self.visible = bool(visible)


class _DummyPyVistaText:
    """Minimal mutable corner annotation used for telemetry assertions."""

    def __init__(self):
        self.updates = []
        self.text_property = _DummyProperty()

    def set_text(self, position, text):
        self.updates.append((position, text))

    def GetTextProperty(self):
        return self.text_property


class _DummyProperty:
    """Minimal mutable VTK/PyVista display property."""

    def __getattr__(self, name):
        if name.startswith("Set"):
            return lambda *_args, **_kwargs: None
        raise AttributeError(name)


class _DummyLegend:
    """Minimal legend actor exposing text and border properties."""

    def __init__(self):
        self.text_property = _DummyProperty()
        self.box_property = _DummyProperty()
        self.position = None

    def GetEntryTextProperty(self):
        return self.text_property

    def GetBoxProperty(self):
        return self.box_property

    def SetPadding(self, *_args):
        return None

    def SetPosition(self, x, y):
        self.position = (x, y)


class _DummyWidgetRepresentation:
    """Minimal VTK widget representation for value and state updates."""

    def __init__(self, value=0):
        self.value = value
        self.state = int(bool(value))

    def SetValue(self, value):
        self.value = value

    def SetState(self, state):
        self.state = state

    def SetColor(self, *_color):
        return None

    def __getattr__(self, name):
        if name.startswith("Set"):
            return lambda *_args, **_kwargs: None
        if name.startswith("Get") and name.endswith("Property"):
            return lambda: self
        raise AttributeError(name)


class _DummyWidget:
    """Minimal widget exposing its mutable representation."""

    def __init__(self, value=0):
        self.representation = _DummyWidgetRepresentation(value)

    def GetRepresentation(self):
        return self.representation


class _DummyCamera:
    """Minimal camera recording chart-space framing changes."""

    def __init__(self):
        self.zoom_factors = []
        self.window_centers = []

    def zoom(self, factor):
        self.zoom_factors.append(factor)

    def SetWindowCenter(self, x, y):
        self.window_centers.append((x, y))


class _DummyChartPlot:
    """Minimal mutable chart line used for cursor assertions."""

    def __init__(self, x, y, **kwargs):
        self.x = list(x)
        self.y = list(y)
        self.options = kwargs

    def update(self, x, y):
        self.x = list(x)
        self.y = list(y)


class _DummyChartAxis:
    """Minimal chart axis accepting PyVista styling attributes."""

    def __init__(self):
        self.label = ""
        self.label_size = 0
        self.tick_label_size = 0


class _DummyChart:
    """Minimal PyVista Chart2D replacement."""

    def __init__(self, **kwargs):
        self.options = kwargs
        self.lines = []
        self.x_axis = _DummyChartAxis()
        self.y_axis = _DummyChartAxis()
        self.title = ""
        self.background_color = None
        self.border_color = None

    def line(self, x, y, **kwargs):
        line = _DummyChartPlot(x, y, **kwargs)
        self.lines.append(line)
        return line


class _DummyTexture:
    """Minimal texture supporting explicit north-up image correction."""

    def __init__(self):
        self.flipped_y = False

    def flip_y(self):
        self.flipped_y = True
        return self


class _DummyPlotter:
    """Minimal plotter mock for non-interactive animation tests."""

    def __init__(self, *_args, **_kwargs):
        self.mesh_options = []
        self.meshes = []
        self.text_actors = []
        self.closed = False
        self.options = _kwargs
        self.control_types = []
        self.point_options = []
        self.point_labels = []
        self.timer_callback = None
        self.text_options = []
        self.legend_options = []
        self.legend_actors = []
        self.background_calls = []
        self.show_options = None
        self.shadows_enabled = False
        self.charts = []
        self.camera_position = None
        self.camera = _DummyCamera()
        self.opened_gif = None
        self.opened_movie = None
        self.frame_count = 0
        self.image_transparent_background = False
        self.axes_options = []

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append(mesh)
        self.mesh_options.append(kwargs)
        return _DummyPyVistaMesh()

    def add_axes(self, **kwargs):
        self.axes_options.append(kwargs)

    def add_text(self, *_args, **_kwargs):
        text = _DummyPyVistaText()
        self.text_actors.append(text)
        self.text_options.append(_kwargs)
        return text

    def add_legend(self, **kwargs):
        self.legend_options.append(kwargs)
        legend = _DummyLegend()
        self.legend_actors.append(legend)
        return legend

    def set_background(self, color, **kwargs):
        self.background_calls.append((color, kwargs.get("top")))

    def add_points(self, _points, **kwargs):
        self.point_options.append(kwargs)
        return _DummyPyVistaMesh()

    def add_point_labels(self, _points, labels, **_kwargs):
        self.point_labels.extend(labels)
        return _DummyPyVistaText()

    def add_slider_widget(self, callback, _rng, value=None, **_kwargs):
        self.control_types.append("timeline")
        widget = _DummyWidget(value)
        callback(value)
        return widget

    def add_text_slider_widget(self, callback, data, value=None, **_kwargs):
        self.control_types.append("speed")
        widget = _DummyWidget(value)
        callback(data[value])
        return widget

    def add_checkbox_button_widget(self, callback, value=False, **_kwargs):
        self.control_types.append("play")
        self.play_callback = callback
        return _DummyWidget(value)

    def add_timer_event(self, _max_steps=None, _duration=None, callback=None, **kwargs):
        self.timer_callback = callback or kwargs["callback"]

    def add_chart(self, chart):
        self.charts.append(chart)

    def open_gif(self, filename, **kwargs):
        self.opened_gif = (filename, kwargs)

    def open_movie(self, filename, **kwargs):
        self.opened_movie = (filename, kwargs)

    def write_frame(self):
        self.frame_count += 1

    def screenshot(self, **_kwargs):
        frame = np.zeros((2, 2, 4), dtype=np.uint8)
        frame[..., :3] = 128
        frame[0, 0, 3] = 255
        return frame

    def enable_shadows(self):
        self.shadows_enabled = True

    def show(self, **kwargs):
        self.show_options = kwargs

    def __getattr__(self, name):
        if name.startswith(("add_", "enable_", "set_", "show_")) or name in {
            "show",
            "update",
            "view_isometric",
            "reset_camera",
        }:
            return lambda *_args, **_kwargs: None
        raise AttributeError(name)

    def close(self):
        self.closed = True


def _mock_pyvista_module(monkeypatch):
    """Install a minimal PyVista module in ``sys.modules`` for tests."""

    pyvista_module = types.ModuleType("pyvista")
    pyvista_module.created_plotters = []

    def create_plotter(*args, **kwargs):
        plotter = _DummyPlotter(*args, **kwargs)
        pyvista_module.created_plotters.append(plotter)
        return plotter

    pyvista_module.Plotter = create_plotter
    pyvista_module.PolyData = _DummyPyVistaMesh
    pyvista_module.Chart2D = _DummyChart
    pyvista_module.read = lambda *_args, **_kwargs: _DummyPyVistaMesh()
    pyvista_module.texture_paths = []

    def read_texture(path):
        pyvista_module.texture_paths.append(path)
        texture = _DummyTexture()
        pyvista_module.last_texture = texture
        return texture

    pyvista_module.read_texture = read_texture
    pyvista_module.Arrow = lambda *_args, **_kwargs: _DummyPyVistaMesh()
    pyvista_module.plane_calls = []

    def create_plane(*args, **kwargs):
        pyvista_module.plane_calls.append((args, kwargs))
        return _DummyPyVistaMesh()

    pyvista_module.Plane = create_plane
    pyvista_module.sphere_calls = []

    def create_sphere(*args, **kwargs):
        pyvista_module.sphere_calls.append((args, kwargs))
        return _DummyPyVistaMesh()

    pyvista_module.Sphere = create_sphere
    monkeypatch.setitem(sys.modules, "pyvista", pyvista_module)
    return pyvista_module


def test_animate_trajectory_builds_informative_pyvista_scene(
    flight_calisto, monkeypatch
):
    """Test flight trajectory animation entry point through the plots layer."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    result = flight_calisto.plots.animate_trajectory(
        start=0.0,
        stop=flight_calisto.t_final,
        time_step=flight_calisto.t_final / 20,
        window_size=(900, 600),
    )

    # Assert
    assert result is None
    plotter = pyvista.created_plotters[0]
    labels = {options.get("label") for options in plotter.mesh_options}
    assert "Ground projection" in labels
    assert "Rocket (not to scale)" in labels
    assert plotter.text_actors[0].updates
    assert plotter.text_options[0]["font_size"] == 9
    assert plotter.legend_options[0]["size"] == (0.145, 0.115)
    assert plotter.closed
    assert plotter.options["window_size"] == (900, 600)
    assert plotter.control_types == ["timeline", "speed", "play"]
    assert {options["point_size"] for options in plotter.point_options} == {9, 16}
    assert len(plotter.point_options) == 2 * len(plotter.point_labels)
    assert {"Start", "Motor burnout", "Apogee", "End"} <= set(plotter.point_labels)
    for _trigger_time, parachute in flight_calisto.parachute_events:
        assert f"{parachute.name} trigger" in plotter.point_labels
        assert f"{parachute.name} open" in plotter.point_labels
    colored_overlays = [
        options
        for options in plotter.mesh_options
        if options.get("label")
        in {
            "Simulated path",
            "Flown path",
            "Velocity direction",
            "Wind velocity (toward)",
            "Ground projection",
        }
    ]
    assert all(options["lighting"] is False for options in colored_overlays)
    simulated_path = next(
        options
        for options in plotter.mesh_options
        if options.get("label") == "Simulated path"
    )
    assert "scalars" not in simulated_path
    assert "cmap" not in simulated_path
    assert simulated_path["color"] == "#5B6573"
    assert simulated_path["line_width"] == pytest.approx(1.8)
    flown_path = next(
        options
        for options in plotter.mesh_options
        if options.get("label") == "Flown path"
    )
    assert flown_path["scalars"] == "Speed (m/s)"
    assert flown_path["cmap"] == "viridis"
    assert flown_path["show_scalar_bar"] is True
    assert flown_path["line_width"] == pytest.approx(4)
    simulated_index = plotter.mesh_options.index(simulated_path)
    simulated_mesh = plotter.meshes[simulated_index]
    dashed_cells = np.asarray(simulated_mesh.lines).reshape((-1, 3))
    assert np.all(dashed_cells[:, 0] == 2)
    assert np.all(dashed_cells[:, 2] - dashed_cells[:, 1] == 1)
    assert len(simulated_mesh.point_data["Speed (m/s)"]) == len(simulated_mesh.points)
    assert "render_lines_as_tubes" not in flown_path
    telemetry_lines = plotter.text_actors[0].updates[-1][1].splitlines()
    assert len({len(line) for line in telemetry_lines}) == 1


def test_animate_rotate_builds_body_axis_reference_scene(flight_calisto, monkeypatch):
    """Test flight rotation animation entry point through the plots layer."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    result = flight_calisto.plots.animate_rotate(
        start=0.0,
        stop=0.001,
        time_step=0.001,
    )

    # Assert
    assert result is None
    labels = {
        options.get("label") for options in pyvista.created_plotters[0].mesh_options
    }
    assert {"Body X — pitch", "Body Y — yaw", "Body Z — roll"} <= labels
    assert {"Velocity direction", "Wind velocity (toward)"} <= labels
    assert {"Rocket", "Inertial up", "Inertial +U pole"}.isdisjoint(labels)
    assert pyvista.created_plotters[0].control_types == [
        "timeline",
        "speed",
        "play",
    ]
    assert pyvista.created_plotters[0].text_options[0]["font_size"] == 8
    assert pyvista.created_plotters[0].axes_options[0]["viewport"] == (
        0.20,
        0.09,
        0.33,
        0.22,
    )
    legend = pyvista.created_plotters[0].legend_options[0]
    assert legend["size"] == (0.145, 0.105)
    legend_labels = {label for label, _color in legend["labels"]}
    assert legend_labels == {
        "Body X — pitch",
        "Body Y — yaw",
        "Body Z — roll",
        "Velocity direction",
        "Wind velocity (toward)",
    }


def test_animation_color_scheme_drives_actor_and_legend_colors(
    flight_calisto, monkeypatch
):
    """Test that scene and legend colors share the centralized scheme."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)
    colors = flight_calisto.plots._animation_color_scheme()

    # Act
    flight_calisto.plots.animate_rotate(
        start=0.0,
        stop=0.001,
        time_step=0.001,
    )

    # Assert
    assert colors["velocity"] == "#E69F00"
    plotter = pyvista.created_plotters[0]
    actor_colors = {
        options.get("label"): options.get("color") for options in plotter.mesh_options
    }
    assert actor_colors["Body X — pitch"] == colors["body_x"]
    assert actor_colors["Velocity direction"] == colors["velocity"]
    legend_colors = dict(plotter.legend_options[0]["labels"])
    assert legend_colors["Body X — pitch"] == colors["body_x"]
    assert legend_colors["Velocity direction"] == colors["velocity"]


def test_animate_trajectory_accepts_reserved_visualization_kwargs(
    flight_calisto, monkeypatch
):
    """Test reserved kwargs, external rendering, texture and line styling."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    flight_calisto.plots.animate_trajectory(
        start=0.0,
        stop=0.001,
        time_step=0.001,
        background_color="#AACCEE",
        playback_controls=False,
        show_subsatellite_point=False,
        ground_image="ground.png",
        backend="client",
        force_external=True,
        shadows=True,
        trajectory_line_width=10,
    )

    # Assert
    plotter = pyvista.created_plotters[0]
    assert plotter.control_types == []
    assert plotter.options["notebook"] is False
    assert plotter.options["off_screen"] is False
    assert "background_color" not in plotter.options
    assert plotter.show_options == {
        "auto_close": False,
        "jupyter_backend": "none",
    }
    assert plotter.shadows_enabled
    assert pyvista.texture_paths == ["ground.png"]
    labels = {options.get("label") for options in plotter.mesh_options}
    assert "Ground projection" not in labels
    widths = {
        options.get("label"): options.get("line_width")
        for options in plotter.mesh_options
    }
    assert widths["Simulated path"] == pytest.approx(4.5)
    assert widths["Flown path"] == pytest.approx(10)


def test_animate_trajectory_adds_kinematic_charts_and_follow_camera(
    flight_calisto, monkeypatch
):
    """Test optional SI kinematic histories and trajectory camera tracking."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    flight_calisto.plots.animate_trajectory(
        start=0.0,
        stop=0.01,
        time_step=0.01,
        color_by=False,
        show_kinematic_plots=True,
        camera_mode="follow",
        playback_controls=False,
    )

    # Assert
    plotter = pyvista.created_plotters[0]
    assert [chart.title for chart in plotter.charts] == [
        "Altitude AGL (m)",
        "Speed (m/s)",
        "Acceleration (m/s²)",
    ]
    assert all(chart.options["size"] == (0.312, 0.243) for chart in plotter.charts)
    assert [chart.options["loc"] for chart in plotter.charts] == [
        (0.01, 0.12),
        (0.01, 0.38),
        (0.01, 0.64),
    ]
    assert plotter.text_options[0]["position"] == "upper_right"
    assert plotter.text_actors[0].updates[-1][0] == "upper_right"
    assert plotter.legend_options[0]["loc"] == "lower right"
    assert plotter.legend_actors[0].position == pytest.approx((0.815, 0.225))
    assert plotter.axes_options[0]["viewport"] == (0.63, 0.09, 0.76, 0.22)
    assert plotter.camera_position is not None
    path_options = {options.get("label"): options for options in plotter.mesh_options}
    assert path_options["Simulated path"]["color"]
    assert "scalars" not in path_options["Simulated path"]


def test_animate_trajectory_places_latlon_ground_image_in_enu(
    flight_calisto, monkeypatch
):
    """Test local equirectangular placement of latitude/longitude imagery."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)
    latitude = float(flight_calisto.env.latitude)
    longitude = float(flight_calisto.env.longitude)
    bounds = (longitude - 0.01, longitude + 0.01, latitude - 0.01, latitude + 0.01)

    # Act
    flight_calisto.plots.animate_trajectory(
        start=0.0,
        stop=0.001,
        time_step=0.001,
        ground_image={
            "image": "satellite.png",
            "bounds": bounds,
            "coordinates": "latlon",
            "flip_y": True,
        },
    )

    # Assert
    plane_options = pyvista.plane_calls[0][1]
    earth_radius = 6_371_000.0
    expected_width = (
        earth_radius * np.cos(np.radians(latitude)) * np.radians(bounds[1] - bounds[0])
    )
    expected_height = earth_radius * np.radians(bounds[3] - bounds[2])
    assert plane_options["i_size"] == pytest.approx(expected_width)
    assert plane_options["j_size"] == pytest.approx(expected_height)
    assert plane_options["center"][:2] == pytest.approx((0, 0), abs=1e-8)
    assert pyvista.last_texture.flipped_y


def test_animate_rotate_adds_diagnostics_and_stability_markers(
    flight_calisto, monkeypatch
):
    """Test optional kinematic, attitude, CM and CP scientific overlays."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    flight_calisto.plots.animate_rotate(
        start=0.0,
        stop=0.01,
        time_step=0.01,
        show_kinematic_plots=True,
        show_attitude_plots=True,
        show_cp_cm=True,
        camera_mode="body",
        playback_controls=False,
    )

    # Assert
    plotter = pyvista.created_plotters[0]
    assert len(plotter.charts) == 6
    assert {chart.title for chart in plotter.charts} >= {
        "Aerodynamic angles (deg)",
        "3-1-3 Euler angles (deg)",
        "Body angular rates (deg/s)",
    }
    labels = {options.get("label") for options in plotter.mesh_options}
    assert {"Center of mass", "Center of pressure"} <= labels
    assert plotter.camera_position is not None
    assert plotter.text_options[0]["position"] == "upper_left"
    assert plotter.text_actors[0].updates[-1][0] == "upper_left"
    assert plotter.legend_options[0]["loc"] == "upper center"
    assert [chart.options["size"] for chart in plotter.charts] == [
        (0.247, 0.243),
    ] * 6
    assert [chart.options["loc"][0] for chart in plotter.charts] == [
        0.01,
        0.01,
        0.01,
        0.743,
        0.743,
        0.743,
    ]
    marker_centers = [
        np.asarray(kwargs["center"])
        for _args, kwargs in pyvista.sphere_calls
        if "center" in kwargs
    ]
    assert len(marker_centers) >= 4
    body_axis = flight_calisto.plots._animation_transformation(0.0)[:3, 2]
    assert all(
        np.linalg.norm(np.cross(marker_center, body_axis)) > 0
        for marker_center in marker_centers[:2]
    )


def test_animate_trajectory_exports_deterministic_gif_with_camera_path(
    flight_calisto, monkeypatch, tmp_path
):
    """Test fixed-frame export, output resolution and camera interpolation."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)
    export_file = tmp_path / "flight.gif"
    camera_path = [
        [(1, 0, 0), (0, 0, 0), (0, 0, 1)],
        [(2, 0, 0), (0, 1, 0), (0, 0, 1)],
    ]

    # Act
    result = flight_calisto.plots.animate_trajectory(
        start=0.0,
        stop=0.1,
        time_step=0.1,
        export_file=export_file,
        export_fps=10,
        export_resolution=(640, 360),
        transparent_background=True,
        camera_path=camera_path,
    )

    # Assert
    plotter = pyvista.created_plotters[0]
    assert result == os.fspath(export_file)
    assert plotter.options["window_size"] == (640, 360)
    assert plotter.options["off_screen"] is True
    assert plotter.opened_gif is None
    assert plotter.frame_count == 0
    assert plotter.image_transparent_background
    assert plotter.camera_position == camera_path[-1]
    assert plotter.control_types == []
    assert export_file.read_bytes().startswith(b"GIF")


def test_animate_rotate_merges_color_scheme_override(flight_calisto, monkeypatch):
    """Test public palette overrides in both actor and legend styling."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    flight_calisto.plots.animate_rotate(
        start=0.0,
        stop=0.001,
        time_step=0.001,
        color_scheme={"velocity": "#123456"},
    )

    # Assert
    plotter = pyvista.created_plotters[0]
    velocity_actor = next(
        options
        for options in plotter.mesh_options
        if options.get("label") == "Velocity direction"
    )
    assert velocity_actor["color"] == "#123456"
    assert dict(plotter.legend_options[0]["labels"])["Velocity direction"] == "#123456"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"color_by": "temperature"}, "color_by must be"),
        (
            {"export_file": "flight.mp4", "transparent_background": True},
            "supported for GIF",
        ),
        ({"color_scheme": {"unknown": "red"}}, "Unknown color_scheme keys"),
    ],
)
def test_animation_rejects_invalid_advanced_options(
    flight_calisto, monkeypatch, kwargs, message
):
    """Test errors for unsupported scalar, export and palette settings."""

    # Arrange
    _mock_pyvista_module(monkeypatch)

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        flight_calisto.plots.animate_trajectory(
            start=0.0,
            stop=0.001,
            time_step=0.001,
            **kwargs,
        )


def test_animation_scalars_are_finite(flight_calisto):
    """Test all supported trajectory scalars against Flight functions."""

    # Arrange
    time_value = min(0.1, flight_calisto.t_final)
    color_modes = [
        "speed",
        "mach",
        "dynamic_pressure",
        "acceleration",
        "altitude",
    ]

    # Act
    values = [
        flight_calisto.plots._animation_scalar(time_value, color_by)
        for color_by in color_modes
    ]

    # Assert
    assert np.all(np.isfinite(values))


def test_animate_rotate_forwards_selected_backend(flight_calisto, monkeypatch):
    """Test explicit notebook backend forwarding without external override."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    flight_calisto.plots.animate_rotate(
        start=0.0,
        stop=0.001,
        time_step=0.001,
        backend="trame",
    )

    # Assert
    assert pyvista.created_plotters[0].show_options["jupyter_backend"] == "trame"


def test_animate_rotate_legend_position_when_both_plots_active(
    flight_calisto, monkeypatch
):
    """Test that the legend is placed in the top middle (upper center) when both
    kinematic and attitude plots are enabled.
    """

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    flight_calisto.plots.animate_rotate(
        start=0.0,
        stop=0.001,
        time_step=0.001,
        show_kinematic_plots=True,
        show_attitude_plots=True,
    )

    # Assert
    plotter = pyvista.created_plotters[0]
    assert plotter.legend_options[0]["loc"] == "upper center"


def test_animate_rotate_legend_position_default(flight_calisto, monkeypatch):
    """Test that the legend is placed in the default upper right position by default."""

    # Arrange
    pyvista = _mock_pyvista_module(monkeypatch)

    # Act
    flight_calisto.plots.animate_rotate(
        start=0.0,
        stop=0.001,
        time_step=0.001,
    )

    # Assert
    plotter = pyvista.created_plotters[0]
    assert plotter.legend_options[0]["loc"] == "upper right"


@pytest.mark.parametrize("method_name", ["animate_trajectory", "animate_rotate"])
def test_animation_rejects_unknown_backend(flight_calisto, monkeypatch, method_name):
    """Test validation of RocketPy's visualization backend option."""

    # Arrange
    _mock_pyvista_module(monkeypatch)
    animation = getattr(flight_calisto.plots, method_name)

    # Act / Assert
    with pytest.raises(ValueError, match="Invalid backend"):
        animation(
            start=0.0,
            stop=0.001,
            time_step=0.001,
            backend="invalid",
        )


def test_animate_trajectory_raises_when_pyvista_is_missing(flight_calisto, monkeypatch):
    """Test that an informative ImportError is raised when PyVista is unavailable."""

    # Arrange
    real_import = builtins.__import__

    def import_without_pyvista(name, *args, **kwargs):
        if name == "pyvista" or name.startswith("pyvista."):
            raise ImportError("No module named 'pyvista'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pyvista", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_pyvista)

    # Act / Assert
    with pytest.raises(ImportError, match="optional dependency"):
        flight_calisto.plots.animate_trajectory(
            start=0.0,
            stop=0.001,
            time_step=0.001,
        )


def test_animate_rotate_raises_when_time_range_is_invalid(flight_calisto, monkeypatch):
    """Test validation error for invalid animation time range."""

    # Arrange
    _mock_pyvista_module(monkeypatch)
    # Act / Assert
    with pytest.raises(ValueError, match="Invalid animation time range"):
        flight_calisto.plots.animate_rotate(
            start=1.0,
            stop=0.5,
            time_step=0.1,
        )


def test_animate_trajectory_raises_when_stl_file_is_missing(
    flight_calisto, monkeypatch
):
    """Test file validation when STL path does not exist."""

    # Arrange
    _mock_pyvista_module(monkeypatch)

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Could not find the 3D model file"):
        flight_calisto.plots.animate_trajectory(
            "missing_model.stl",
            start=0.0,
            stop=0.1,
            time_step=0.1,
        )


@pytest.mark.parametrize("invalid_time_step", [0, -0.1])
def test_animate_trajectory_raises_when_time_step_is_non_positive(
    flight_calisto, monkeypatch, invalid_time_step
):
    """Test validation error when animation time_step is not strictly positive."""

    # Arrange
    _mock_pyvista_module(monkeypatch)
    # Act / Assert
    with pytest.raises(ValueError, match="Invalid time_step"):
        flight_calisto.plots.animate_trajectory(
            start=0.0,
            stop=0.1,
            time_step=invalid_time_step,
        )


def test_animate_rotate_raises_when_stop_exceeds_flight_end(
    flight_calisto, monkeypatch
):
    """Test validation error when stop time exceeds available simulation range."""

    # Arrange
    _mock_pyvista_module(monkeypatch)
    # Act / Assert
    with pytest.raises(ValueError, match="Invalid animation time range"):
        flight_calisto.plots.animate_rotate(
            start=0.0,
            stop=flight_calisto.t_final + 0.1,
            time_step=0.1,
        )


def test_animate_trajectory_raises_when_default_model_is_missing(
    flight_calisto, monkeypatch
):
    """Test failure path when default packaged STL model is unavailable."""

    # Arrange
    _mock_pyvista_module(monkeypatch)
    monkeypatch.setattr(
        flight_calisto.plots,
        "_resolve_animation_model_path",
        lambda _file_name: "missing_default_model.stl",
    )

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Could not find the 3D model file"):
        flight_calisto.plots.animate_trajectory(
            start=0.0,
            stop=0.1,
            time_step=0.1,
        )


@pytest.mark.parametrize("method_name", ["animate_trajectory", "animate_rotate"])
def test_animation_raises_when_playback_speed_is_non_positive(
    flight_calisto, monkeypatch, method_name
):
    """Test that wall-clock playback speed must be strictly positive."""

    # Arrange
    _mock_pyvista_module(monkeypatch)
    animation = getattr(flight_calisto.plots, method_name)

    # Act / Assert
    with pytest.raises(ValueError, match="Invalid playback_speed"):
        animation(
            start=0.0,
            stop=0.1,
            time_step=0.1,
            playback_speed=0,
        )


def test_rotation_matrix_from_quaternion_rotates_body_x_to_north(flight_calisto):
    """Test the body-to-inertial quaternion convention used by the meshes."""

    # Arrange
    half_angle = np.sqrt(0.5)

    # Act
    rotation = flight_calisto.plots._rotation_matrix_from_quaternion(
        half_angle, 0, 0, half_angle
    )

    # Assert
    assert rotation[:3, :3] @ np.array([1, 0, 0]) == pytest.approx([0, 1, 0])


def test_safe_unit_vector_uses_up_direction_for_zero_velocity(flight_calisto):
    """Test that a stationary launch state produces a valid PyVista arrow."""

    # Arrange
    zero_velocity = np.zeros(3)

    # Act
    direction = flight_calisto.plots._safe_unit_vector(zero_velocity)

    # Assert
    assert direction == pytest.approx([0, 0, 1])
    assert np.all(np.isfinite(direction))


@pytest.mark.parametrize(
    ("launch_hour", "expected_bottom"),
    [
        (6, (143 / 255, 179 / 255, 201 / 255)),
        (19, (143 / 255, 179 / 255, 201 / 255)),
        (20, (10 / 255, 15 / 255, 20 / 255)),
        (2, (10 / 255, 15 / 255, 20 / 255)),
    ],
)
def test_animation_background_palette_uses_local_launch_hour(
    flight_calisto, monkeypatch, launch_hour, expected_bottom
):
    """Test the intentionally simple 06:00-to-20:00 daylight rule."""

    # Arrange
    monkeypatch.setattr(
        flight_calisto.env,
        "local_date",
        types.SimpleNamespace(hour=launch_hour),
    )

    # Act
    palette = flight_calisto.plots._animation_background_palette()

    # Assert
    assert palette["launch_bottom"] == pytest.approx(expected_bottom)


def test_animation_background_blends_custom_color_to_space(flight_calisto):
    """Test custom launch color preservation and the 50 km transition cap."""

    # Arrange
    palette = flight_calisto.plots._animation_background_palette("#6699CC")

    # Act
    launch_bottom, launch_top = flight_calisto.plots._animation_background_at_altitude(
        palette, 0
    )
    space_bottom, space_top = flight_calisto.plots._animation_background_at_altitude(
        palette, 50_000
    )

    # Assert
    expected_launch = (0.4, 0.6, 0.8)
    assert launch_bottom == pytest.approx(expected_launch)
    assert launch_top == pytest.approx(expected_launch)
    assert space_bottom == pytest.approx(palette["space_bottom"])
    assert space_top == pytest.approx(palette["space_top"])
