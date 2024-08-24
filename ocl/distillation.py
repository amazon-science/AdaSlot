import copy
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from ocl import scheduling
from ocl.utils.routing import Combined
from ocl.utils.trees import get_tree_element


class EMASelfDistillation(nn.Module):
    def __init__(
        self,
        student: Union[nn.Module, Dict[str, nn.Module]],
        schedule: scheduling.HPScheduler,
        student_remapping: Optional[Dict[str, str]] = None,
        teacher_remapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        # Do this for convenience to reduce crazy amount of nesting.
        if isinstance(student, dict):
            student = Combined(student)
        if student_remapping is None:
            student_remapping = {}
        if teacher_remapping is None:
            teacher_remapping = {}

        self.student = student
        self.teacher = copy.deepcopy(student)
        self.schedule = schedule
        self.student_remapping = {key: value.split(".") for key, value in student_remapping.items()}
        self.teacher_remapping = {key: value.split(".") for key, value in teacher_remapping.items()}

    def build_input_dict(self, inputs, remapping):
        if not remapping:
            return inputs
        # This allows us to bing the initial input and previous_output into a similar format.
        output_dict = {}
        for output_path, input_path in remapping.items():
            source = get_tree_element(inputs, input_path)

            output_path = output_path.split(".")
            cur_search = output_dict
            for path_part in output_path[:-1]:
                # Iterate along path and create nodes that do not exist yet.
                try:
                    # Get element prior to last.
                    cur_search = get_tree_element(cur_search, [path_part])
                except ValueError:
                    # Element does not yet exist.
                    cur_search[path_part] = {}
                    cur_search = cur_search[path_part]

            cur_search[output_path[-1]] = source
        return output_dict

    def forward(self, inputs: Dict[str, Any]):
        if self.training:
            with torch.no_grad():
                m = self.schedule(inputs["global_step"])  # momentum parameter
                for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # prefix variable similar to combined module.
        prefix: List[str]
        if "prefix" in inputs.keys():
            prefix = inputs["prefix"]
        else:
            prefix = []
            inputs["prefix"] = prefix

        outputs = get_tree_element(inputs, prefix)

        # Forward pass student.
        prefix.append("student")
        outputs["student"] = {}
        student_inputs = self.build_input_dict(inputs, self.student_remapping)
        outputs["student"] = self.student(inputs={**inputs, **student_inputs})
        # Teacher and student share the same code, thus paths also need to be the same.  To ensure
        # that we save the student outputs and run the teacher as if it where the student.
        student_output = outputs["student"]

        # Forward pass teacher, but pretending to be student.
        outputs["student"] = {}
        teacher_inputs = self.build_input_dict(inputs, self.teacher_remapping)

        with torch.no_grad():
            outputs["teacher"] = self.teacher(inputs={**inputs, **teacher_inputs})
        prefix.pop()

        # Set correct outputs again.
        outputs["student"] = student_output

        return outputs
