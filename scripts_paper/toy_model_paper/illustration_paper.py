from typing import Dict, Hashable, List
import matplotlib.pyplot as plt
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import plot_resource_individual_gantt_preemptive, plot_task_gantt
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, MS_RCPSPModel_Variant, \
    MS_RCPSPSolution_Preemptive, MS_RCPSPSolution_Preemptive_Variant, Employee, SkillDetail, \
    SpecialConstraintsDescription
import numpy as np

from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE


def build_toy_problem():
    horizon = 50
    tasks = ["A0", "A1", "A2", "A3", "A4", "A5"]
    skills = ["l1", "l2", "l3", "l4"]
    employees_list = ["O1", "O2"]
    ressources_list = ["R1"]
    mode_details = {"A0": {1: {"duration": 0}},
                    "A1": {1: {"duration": 5, "R1": 1, "l1": 1}},
                    "A2": {1: {"duration": 1, "R1": 1, "l3": 1, "l4": 1}},
                    "A3": {1: {"duration": 3, "R1": 1, "l2": 1}},
                    "A4": {1: {"duration": 2, "l3": 1}},
                    "A5": {1: {"duration": 0}}}
    partial_preemption_data:  Dict[Hashable, Dict[int, Dict[str, bool]]] = {"A0": {1: {"R1": True}},
                                                                            "A1": {1: {"R1": True}},
                                                                            "A2": {1: {"R1": True}},
                                                                            "A3": {1: {"R1": False}},
                                                                            "A4": {1: {"R1": True}},
                                                                            "A5": {1: {"R1": True}}}
    preemptive_indicator = {"A0": False, "A1": True, "A2": False, "A3": True, "A4": False, "A5": False}
    employees = {"O1": Employee(dict_skill={"l1": SkillDetail(1, 1, 1),
                                            "l3": SkillDetail(1, 1, 1)},
                                calendar_employee=[True]*horizon),
                 "02": Employee(dict_skill={"l1": SkillDetail(1, 1, 1),
                                            "l2": SkillDetail(1, 1, 1),
                                            "l4": SkillDetail(1, 1, 1)},
                                calendar_employee=[True]*horizon)}
    starting_times_window = {"A2": (2, None), "A4": (5, None)}
    end_time_window = {"A2": (None, 3)}
    successors: Dict[str, List[str]] = {"A0": ["A" + str(i)
                                               for i in range(1, 6)],
                                        "A1": ["A5"],
                                        "A2": ["A5"],
                                        "A3": ["A5"],
                                        "A4": ["A5"],
                                        "A5": []}
    special_constraints = SpecialConstraintsDescription(start_times_window=starting_times_window,
                                                        end_times_window=end_time_window)
    return MS_RCPSPModel(skills_set=set(skills),
                         resources_set=set(ressources_list),
                         non_renewable_resources=set(),
                         resources_availability={"R1": 2*np.ones(horizon)},
                         employees=employees,
                         employees_availability=None,
                         mode_details=mode_details,
                         successors=successors,
                         horizon=50,
                         tasks_list=tasks,
                         source_task="A0",
                         sink_task="A5",
                         preemptive=True,
                         preemptive_indicator=preemptive_indicator,
                         special_constraints=special_constraints,
                         partial_preemption_data=partial_preemption_data)


def run_toy():
    model = build_toy_problem_full_variant()
    cp_solver = CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE(rcpsp_model=model)
    cp_solver.init_model(max_time=100,
                         max_preempted=10,
                         nb_preemptive=10,
                         possibly_preemptive=[model.preemptive_indicator.get(t, True)
                                              for t in model.tasks_list],
                         partial_solution=model.special_constraints,
                         add_partial_solution_hard_constraint=True,
                         strictly_disjunctive_subtasks=False,
                         unit_usage_preemptive=True)
    result_storage = cp_solver.solve(parameters_cp=ParametersCP.default())
    best_sol, fit = result_storage.get_last_best_solution()
    best_sol: MS_RCPSPSolution_Preemptive_Variant = best_sol
    print("Modes found = ", best_sol.modes)
    print(model.satisfy(best_sol))
    #fig = plot_resource_individual_gantt_preemptive(rcpsp_model=model,
    #                                                rcpsp_sol=best_sol,
    #                                                title_figure="Schedule per operator",
    #                                                annotate_all_subparts=True, one_color_per_task=True)
    #fig2 = plot_task_gantt(rcpsp_model=model,
    #                       rcpsp_sol=best_sol,
    #                       annotate_all_subparts=True,
    #                       one_color_per_task=True)
    #fig2.savefig("solution_task_gantt.png")
    #fig.savefig("individual_gantt.png")
    from plot_with_ressource_partial_preemption import plot_resource_individual_gantt_preemptive_with_ressource

    fig3, ax3 = plt.subplots(2,
                             figsize=(11, 3))
    fig3 = plot_resource_individual_gantt_preemptive_with_ressource(rcpsp_model=model,
                                                                    rcpsp_sol=best_sol,
                                                                    title_figure="",
                                                                    # "Schedule per operator and resource",
                                                                    annotate_all_subparts=True,
                                                                    use_grid=False,
                                                                    one_color_per_task=True,
                                                                    fig=fig3, ax=ax3)
    plt.tight_layout()
    fig3.savefig("ind_gantt_bis.png", dpi=1000, bbox_inches="tight")
    plt.show()


def build_toy_problem_full_variant():
    horizon = 50
    tasks = ["A0", "A1", "A2", "A3", "A4", "A5"]
    skills = ["l1", "l2", "l3", "l4"]
    employees_list = ["O1", "O2"]
    ressources_list = ["R1"]
    mode_details = {"A0": {1: {"duration": 0}},
                    "A1": {1: {"duration": 5, "R1": 1, "l1": 1},
                           2: {"duration": 3, "R1": 2, "l1": 1}},
                    "A2": {1: {"duration": 1, "R1": 1, "l3": 1, "l4": 1},
                           2: {"duration": 2, "R1": 1, "l3": 1}},
                    "A3": {1: {"duration": 3, "R1": 1, "l2": 1},
                           2: {"duration": 2, "R1": 1, "l3": 1, "l2": 1}},
                    "A4": {1: {"duration": 2, "l3": 1}, 2: {"duration": 3, "R1": 1}},
                    "A5": {1: {"duration": 0}}}
    partial_preemption_data:  Dict[Hashable, Dict[int, Dict[str, bool]]] = {"A0": {1: {"R1": True}},
                                                                            "A1": {1: {"R1": False},
                                                                                   2: {"R1": False}},
                                                                            "A2": {1: {"R1": True},
                                                                                   2: {"R1": True}},
                                                                            "A3": {1: {"R1": False},
                                                                                   2: {"R1": False}},
                                                                            "A4": {1: {"R1": True},
                                                                                   2: {"R1": True}},
                                                                            "A5": {1: {"R1": True}}}
    preemptive_indicator = {"A0": False, "A1": True, "A2": False,
                            "A3": True, "A4": True, "A5": False}
    employees = {"O-1": Employee(dict_skill={"l1": SkillDetail(1, 1, 1),
                                             "l3": SkillDetail(1, 1, 1)},
                                 calendar_employee=[j % 10 <= 5 for j in range(horizon)]),
                 "O-2": Employee(dict_skill={"l1": SkillDetail(1, 1, 1),
                                             "l2": SkillDetail(1, 1, 1),
                                             "l4": SkillDetail(1, 1, 1)},
                                 calendar_employee=[j % 10 <= 5 for j in range(horizon)])}
    for i in [2, 3, 11, 12]:
        employees["O-2"].calendar_employee[i] = False
    for i in [12, 13, 14, 1, 2]:
        employees["O-1"].calendar_employee[i] = False
    print("Calendars")
    print(employees["O-1"].calendar_employee[:15])
    print(employees["O-2"].calendar_employee[:15])

    starting_times_window = {"A2": (2, None), "A4": (5, None)}
    end_time_window = {"A2": (None, 5)}
    successors: Dict[str, List[str]] = {"A0": ["A" + str(i)
                                               for i in range(1, 6)],
                                        "A1": ["A5"],
                                        "A2": ["A3", "A5"],
                                        "A3": ["A5"],
                                        "A4": ["A5"],
                                        "A5": []}
    start_together = [("A3", "A4")]
    special_constraints = SpecialConstraintsDescription(start_times_window=starting_times_window,
                                                        end_times_window=end_time_window,
                                                        start_together=start_together)

    return MS_RCPSPModel(skills_set=set(skills),
                         resources_set=set(ressources_list),
                         non_renewable_resources=set(),
                         resources_availability={"R1": 2*np.ones(horizon)},
                         employees=employees,
                         employees_availability=None,
                         mode_details=mode_details,
                         successors=successors,
                         horizon=50,
                         tasks_list=tasks,
                         source_task="A0",
                         sink_task="A5",
                         preemptive=True,
                         preemptive_indicator=preemptive_indicator,
                         special_constraints=special_constraints,
                         partial_preemption_data=partial_preemption_data)


if __name__ == "__main__":
    run_toy()