from copy import deepcopy

import numpy as np

from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


_VALID_DESKTOP_VARIANTS = ("clean", "cluttered")
_VALID_RESET_REGIONS = ("small", "large")
_RESET_BOUNDS = {
    "small": {
        "x_range": (-0.08, 0.08),
        "y_range": (-0.08, 0.08),
    },
    "large": {
        "x_range": (-0.16, 0.16),
        "y_range": (-0.16, 0.16),
    },
}
_CLUTTER_RESET_BOUNDS = {
    "x_range": (-0.28, 0.28),
    "y_range": (-0.24, 0.24),
}
_CLUTTER_SPECS = (
    {
        "class": BoxObject,
        "name": "clutter_box",
        "kwargs": {
            "size": [0.015, 0.035, 0.012],
            "rgba": [0.1, 0.35, 0.9, 1],
        },
    },
    {
        "class": CylinderObject,
        "name": "clutter_cylinder",
        "kwargs": {
            "size": [0.015, 0.02],
            "rgba": [0.95, 0.7, 0.15, 1],
        },
    },
    {
        "class": BallObject,
        "name": "clutter_ball",
        "kwargs": {
            "size": [0.018],
            "rgba": [0.5, 0.5, 0.5, 1],
        },
    },
)
_CLUTTER_PRIMITIVE_TYPES = {
    BoxObject: "box",
    CylinderObject: "cylinder",
    BallObject: "ball",
}
_CLUTTER_SIZE_DIMS = {
    BoxObject: 3,
    CylinderObject: 2,
    BallObject: 1,
}
_SUCCESS_PREDICATE_VERSION = "robosuite_stack_staged_rewards_v1"
_UNSUPPORTED_ALIAS_KWARGS = ("variant", "variant_name", "registered_variant_name")
_TASK_CUBE_YAW = 0.0


def _camel_variant_name(desktop_variant, reset_region):
    return "StackBlocks{}{}".format(desktop_variant.capitalize(), reset_region.capitalize())


def _validate_range(bounds, key, label):
    if key not in bounds:
        raise ValueError("{} bounds must define '{}'".format(label, key))

    values = bounds[key]
    if len(values) != 2:
        raise ValueError("{} '{}' must contain exactly two values, got {}".format(label, key, values))

    low, high = values
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("{} '{}' must contain finite values, got {}".format(label, key, values))
    if low >= high:
        raise ValueError("{} '{}' lower bound must be less than upper bound, got {}".format(label, key, values))


def _validate_reset_bounds(bounds, label, table_full_size=None):
    for key in ("x_range", "y_range"):
        _validate_range(bounds, key, label)

    if table_full_size is not None:
        for axis, key in enumerate(("x_range", "y_range")):
            table_half = table_full_size[axis] / 2.0
            low, high = bounds[key]
            if low < -table_half or high > table_half:
                raise ValueError(
                    "{} '{}' {} exceeds table half-size {}".format(label, key, bounds[key], table_half)
                )


def _numeric_list(values, label, expected_len=None, positive=False, min_value=None, max_value=None):
    if not isinstance(values, (list, tuple)):
        raise TypeError("{} must be a list or tuple, got {}".format(label, type(values).__name__))
    if expected_len is not None and len(values) != expected_len:
        raise ValueError("{} must contain exactly {} values, got {}".format(label, expected_len, values))

    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("{} values must be numeric, got {}".format(label, values))
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("{} values must be finite, got {}".format(label, values))
        if positive and value <= 0:
            raise ValueError("{} values must be positive, got {}".format(label, values))
        if min_value is not None and value < min_value:
            raise ValueError("{} values must be >= {}, got {}".format(label, min_value, values))
        if max_value is not None and value > max_value:
            raise ValueError("{} values must be <= {}, got {}".format(label, max_value, values))
        result.append(value)
    return result


def _clutter_object_metadata(spec):
    if not isinstance(spec, dict):
        raise TypeError("StackBlocks clutter spec must be a dict, got {}".format(type(spec).__name__))

    missing = [key for key in ("class", "name", "kwargs") if key not in spec]
    if missing:
        raise ValueError("StackBlocks clutter spec missing required key(s): {}".format(missing))

    object_class = spec["class"]
    if object_class not in _CLUTTER_PRIMITIVE_TYPES:
        expected = sorted(cls.__name__ for cls in _CLUTTER_PRIMITIVE_TYPES)
        raise ValueError(
            "StackBlocks clutter spec has unsupported class {}. Expected one of {}".format(object_class, expected)
        )

    name = spec["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("StackBlocks clutter spec name must be a non-empty string, got {}".format(name))

    kwargs = spec["kwargs"]
    if not isinstance(kwargs, dict):
        raise TypeError("StackBlocks clutter spec '{}' kwargs must be a dict".format(name))
    missing_kwargs = [key for key in ("size", "rgba") if key not in kwargs]
    if missing_kwargs:
        raise ValueError("StackBlocks clutter spec '{}' missing kwarg(s): {}".format(name, missing_kwargs))

    label = "StackBlocks clutter spec '{}'".format(name)
    return {
        "name": name,
        "primitive_type": _CLUTTER_PRIMITIVE_TYPES[object_class],
        "class": object_class.__name__,
        "size": _numeric_list(
            kwargs["size"],
            "{} size".format(label),
            expected_len=_CLUTTER_SIZE_DIMS[object_class],
            positive=True,
        ),
        "rgba": _numeric_list(
            kwargs["rgba"],
            "{} rgba".format(label),
            expected_len=4,
            min_value=0,
            max_value=1,
        ),
    }


def _resolve_variant(desktop_variant, reset_region):
    if desktop_variant not in _VALID_DESKTOP_VARIANTS:
        raise ValueError(
            "Unknown desktop_variant '{}'. Expected one of {}".format(desktop_variant, _VALID_DESKTOP_VARIANTS)
        )
    if reset_region not in _VALID_RESET_REGIONS:
        raise ValueError("Unknown reset_region '{}'. Expected one of {}".format(reset_region, _VALID_RESET_REGIONS))

    reset_bounds = deepcopy(_RESET_BOUNDS[reset_region])
    clutter_specs = deepcopy(_CLUTTER_SPECS) if desktop_variant == "cluttered" else ()
    clutter_object_specs = tuple(_clutter_object_metadata(spec) for spec in clutter_specs)
    _validate_reset_bounds(reset_bounds, "StackBlocks reset")
    _validate_reset_bounds(_CLUTTER_RESET_BOUNDS, "StackBlocks clutter")

    return {
        "variant_name": "{}_{}".format(desktop_variant, reset_region),
        "registered_variant_name": _camel_variant_name(desktop_variant, reset_region),
        "desktop_variant": desktop_variant,
        "reset_region": reset_region,
        "reset_bounds": reset_bounds,
        "clutter_reset_bounds": deepcopy(_CLUTTER_RESET_BOUNDS) if clutter_specs else None,
        "clutter_specs": clutter_specs,
        "clutter_object_specs": clutter_object_specs,
    }


def _as_lists(bounds):
    if bounds is None:
        return None
    return {key: list(value) for key, value in bounds.items()}


class StackBlocks(Stack):
    """
    Conservative Stack variant with named clean / cluttered desktop and small / large reset regions.

    Task semantics are identical to Stack: cubeA succeeds when stacked on cubeB.
    """

    def __init__(self, robots, *args, desktop_variant="clean", reset_region="small", **kwargs):
        if kwargs.get("placement_initializer") is not None:
            raise ValueError(
                "StackBlocks variants own their sampler config; placement_initializer must be None."
            )
        # Stack.__init__ accepts placement_initializer as the 12th positional argument after robots.
        if len(args) > 11 and args[11] is not None:
            raise ValueError(
                "StackBlocks variants own their sampler config; placement_initializer must be None."
            )

        self._variant_config = _resolve_variant(desktop_variant=desktop_variant, reset_region=reset_region)
        self.variant_name = self._variant_config["variant_name"]
        self.registered_variant_name = self._variant_config["registered_variant_name"]
        self.desktop_variant = self._variant_config["desktop_variant"]
        self.reset_region = self._variant_config["reset_region"]
        self.reset_bounds = deepcopy(self._variant_config["reset_bounds"])
        self.clutter_reset_bounds = deepcopy(self._variant_config["clutter_reset_bounds"])
        self.clutter_object_specs = [deepcopy(spec) for spec in self._variant_config["clutter_object_specs"]]
        self.clutter_object_names = [spec["name"] for spec in self.clutter_object_specs]
        self.clutter_object_count = len(self.clutter_object_names)
        self.sampler_names = ()
        self.success_predicate_version = _SUCCESS_PREDICATE_VERSION
        self.clutter_objects = []

        super().__init__(robots, *args, **kwargs)

    @staticmethod
    def _validate_unique_object_names(objects):
        names = [obj.name for obj in objects]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError("StackBlocks object names must be unique. Duplicates: {}".format(duplicates))

    def _make_task_objects(self):
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        return [self.cubeA, self.cubeB]

    def _make_clutter_objects(self):
        return [spec["class"](name=spec["name"], **spec["kwargs"]) for spec in self._variant_config["clutter_specs"]]

    def _build_placement_initializer(self, task_objects, clutter_objects):
        all_objects = task_objects + clutter_objects
        self._validate_unique_object_names(all_objects)

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="TaskObjectSampler",
                mujoco_objects=task_objects,
                x_range=self.reset_bounds["x_range"],
                y_range=self.reset_bounds["y_range"],
                rotation=_TASK_CUBE_YAW,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )
        )

        if clutter_objects:
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="ClutterObjectSampler",
                    mujoco_objects=clutter_objects,
                    x_range=self.clutter_reset_bounds["x_range"],
                    y_range=self.clutter_reset_bounds["y_range"],
                    rotation=None,
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                    rng=self.rng,
                )
            )

        self.sampler_names = (self.placement_initializer.name,) + tuple(self.placement_initializer.samplers.keys())

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model.
        """
        # Skip Stack._load_model; StackBlocks supplies its own objects and samplers.
        super(Stack, self)._load_model()

        _validate_reset_bounds(self.reset_bounds, "StackBlocks reset", table_full_size=self.table_full_size)
        if self.clutter_reset_bounds is not None:
            _validate_reset_bounds(
                self.clutter_reset_bounds,
                "StackBlocks clutter",
                table_full_size=self.table_full_size,
            )

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        task_objects = self._make_task_objects()
        self.clutter_objects = self._make_clutter_objects()
        all_objects = task_objects + self.clutter_objects
        self._build_placement_initializer(task_objects=task_objects, clutter_objects=self.clutter_objects)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

    def _reset_internal(self):
        super()._reset_internal()
        self.set_ep_meta(self.get_variant_metadata())

    def get_variant_metadata(self):
        return {
            "variant_name": self.variant_name,
            "registered_variant_name": self.registered_variant_name,
            "desktop_variant": self.desktop_variant,
            "reset_region": self.reset_region,
            "reset_bounds": _as_lists(self.reset_bounds),
            "task_cube_yaw": _TASK_CUBE_YAW,
            "clutter_reset_bounds": _as_lists(self.clutter_reset_bounds),
            "clutter_object_names": list(self.clutter_object_names),
            "clutter_object_count": self.clutter_object_count,
            "clutter_object_specs": deepcopy(self.clutter_object_specs),
            "sampler_names": list(self.sampler_names),
            "success_predicate_version": self.success_predicate_version,
        }

    def get_stack_blocks_metadata(self):
        return self.get_variant_metadata()


class _StackBlocksAliasMixin:
    _locked_desktop_variant = None
    _locked_reset_region = None

    def __init__(self, *args, desktop_variant=None, reset_region=None, **kwargs):
        for key in _UNSUPPORTED_ALIAS_KWARGS:
            if key in kwargs:
                raise TypeError(
                    "{} does not accept '{}'. Use StackBlocks(desktop_variant=..., reset_region=...) "
                    "or the registered alias class name.".format(self.__class__.__name__, key)
                )

        if self._locked_desktop_variant is None or self._locked_reset_region is None:
            raise ValueError("{} is missing locked StackBlocks variant settings".format(self.__class__.__name__))
        if desktop_variant is not None and desktop_variant != self._locked_desktop_variant:
            raise ValueError(
                "{} locks desktop_variant='{}', got '{}'".format(
                    self.__class__.__name__, self._locked_desktop_variant, desktop_variant
                )
            )
        if reset_region is not None and reset_region != self._locked_reset_region:
            raise ValueError(
                "{} locks reset_region='{}', got '{}'".format(
                    self.__class__.__name__, self._locked_reset_region, reset_region
                )
            )

        super().__init__(
            *args,
            desktop_variant=self._locked_desktop_variant,
            reset_region=self._locked_reset_region,
            **kwargs,
        )


class StackBlocksCleanSmall(_StackBlocksAliasMixin, StackBlocks):
    _locked_desktop_variant = "clean"
    _locked_reset_region = "small"


class StackBlocksCleanLarge(_StackBlocksAliasMixin, StackBlocks):
    _locked_desktop_variant = "clean"
    _locked_reset_region = "large"


class StackBlocksClutteredSmall(_StackBlocksAliasMixin, StackBlocks):
    _locked_desktop_variant = "cluttered"
    _locked_reset_region = "small"


class StackBlocksClutteredLarge(_StackBlocksAliasMixin, StackBlocks):
    _locked_desktop_variant = "cluttered"
    _locked_reset_region = "large"
