import os
from typing import Tuple
from discrete_optimization.generic_rcpsp_tools.large_neighborhood_search_scheduling import \
    LargeNeighborhoodSearchScheduling
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import ParamsConstraintBuilder
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPSolution
from discrete_optimization.rcpsp_multiskill.solvers.cp_solver_mspsp_instlib \
    import CP_MSPSP_MZN, ParametersCP, CPSolverName
from discrete_optimization.generic_tools.cp_tools import StatusSolver
from discrete_optimization.generic_rcpsp_tools.neighbor_builder import mix_lot, ObjectiveSubproblem
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_mspsp_parser import get_data_available_mspsp, \
    parse_dzn_file
from discrete_optimization.rcpsp_multiskill.solvers.lns_post_process_rcpsp import PostProMSRCPSPPreemptive
from discrete_optimization.generic_tools.lns_mip import TrivialInitialSolution
import json
import time
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
this_folder = os.path.dirname(os.path.abspath(__file__))
instance_folder = os.path.join(this_folder, "MSPSP-InstLib/instances/")

strategy = ["default_s",
            "start_s",
            "overlap_s",
            "assign_s",
            "contrib_s",
            "assign_Then_start",
            "contrib_Then_start",
            "start_Then_assign",
            "start_Then_contrib",
            "overlap_Then_assign_Then_start",
            "overlap_Then_contrib_Then_start",
            "priority_input_order",
            "priority_smallest",
            "priority_smallest_load",
            "priority_smallest_largest",
            "priority_first_fail"]


def file_aniti():
    path = os.path.join(this_folder, "Cmax.dat")
    results_aniti = []
    with open(path, "r") as f:
        file = f.readlines()
        for j in range(len(file)):
            line = file[j].split()
            results_aniti += [(line[0], int(line[1]))]
    return results_aniti


def compare_results():
    path = os.path.join(this_folder, "Cmax.dat")
    results_aniti = {}
    with open(path, "r") as f:
        file = f.readlines()
        for j in range(len(file)):
            line = file[j].split()
            results_aniti[os.path.basename(line[0])[:-4]] = int(line[1])
    results_folder = os.path.join(this_folder, "ortools-1703/")
    json_files = [os.path.join(this_folder, f) for f in os.listdir(results_folder) if "json" in f]
    my_results = {}
    import json
    for file in json_files:
        res = json.load(open(file, "r"))
        file = res["file"]
        key = os.path.basename(file)[:-4]
        my_results[key] = res["makespan"]
    common_keys = [k for k in my_results if k in results_aniti]
    sorted_keys = sorted(common_keys)
    array_aniti = np.array([results_aniti[k] for k in sorted_keys])
    array_mine = np.array([my_results[k] for k in sorted_keys])
    counts = {"air": 0, "paper": 0, "equality": 0}

    for j in range(len(array_aniti)):
        print(array_aniti[j], array_mine[j])
        if array_mine[j] < array_aniti[j]:
            counts["air"] += 1
        if array_aniti[j] < array_mine[j]:
            counts["paper"] += 1
        if array_aniti[j] == array_mine[j]:
            counts["equality"] += 1
    print(counts)
    print("Instance, LNS-CP, Grasp, % gain")
    for key in common_keys:
        if my_results[key] < results_aniti[key]:
            print(key, ",", my_results[key], ",", results_aniti[key], ",",
                  (results_aniti[key]-my_results[key])/results_aniti[key]*100, "%")


def run_all_benchmark(cpopt: bool = False, name_folder: str = "rerun/",
                      solver: CPSolverName = CPSolverName.ORTOOLS,
                      only_cp: bool = False):
    folder_improved_solution = os.path.join(this_folder, name_folder)
    if not os.path.exists(folder_improved_solution):
        os.makedirs(folder_improved_solution)
    file_paths = get_data_available_mspsp()
    for set in sorted(file_paths):
        for subset in sorted(file_paths[set]):
            if "1b" not in subset:
                continue
            for file in sorted(file_paths[set][subset]):
                if "dzn" not in file:
                    continue
                print("FILE : ", set, subset, file)
                if os.path.basename(file) + '_best_sol.json' in os.listdir(folder_improved_solution)\
                        or os.path.basename(file) + '_best_solution.json' in os.listdir(folder_improved_solution):
                    print("Done")
                    continue
                t = time.time()
                if only_cp:
                    res, status = run_only_cp(file, time_limit=100)
                    t_end = time.time()
                    best_solution: Tuple[MS_RCPSPSolution, float] = res.get_best_solution_fit()
                    dict_ = {"file": file,
                             "solver_cp": "chuffed",
                             "status": str(status.value),
                             "schedule": best_solution[0].schedule,
                             "employee_usage": {t: {emp: list(best_solution[0].employee_usage[t][emp])
                                                    for emp in best_solution[0].employee_usage[t]}
                                                for t in best_solution[0].employee_usage},
                             "runtime": int(t_end - t),
                             "makespan": best_solution[0].problem.evaluate(best_solution[0])["makespan"]}
                    json.dump(dict_,
                              open(os.path.join(folder_improved_solution, os.path.basename(file) + '_best_sol.json'),
                                   "w"),
                              indent=4)
                else:
                    if not cpopt:
                        res = run_lns(file, time_limit=500)
                    else:
                        res = run_lns_cpopt(file, time_limit=500,
                                            solver_lns=solver)
                    t_end = time.time()
                    best_solution: Tuple[MS_RCPSPSolution, float] = res.get_best_solution_fit()
                    dict_ = {"file": file,
                             "solver_lns": solver.name,
                             "schedule": best_solution[0].schedule,
                             "employee_usage": {t: {emp: list(best_solution[0].employee_usage[t][emp])
                                                    for emp in best_solution[0].employee_usage[t]}
                                                for t in best_solution[0].employee_usage},
                             "runtime": int(t_end - t),
                             "makespan": best_solution[0].problem.evaluate(best_solution[0])["makespan"]}
                    json.dump(dict_, open(os.path.join(folder_improved_solution, os.path.basename(file) + '_best_sol.json'),
                                          "w"),
                              indent=4)
            else:
                pass


def run_lns(file=None, time_limit=1000):
    model = parse_dzn_file(file_path=file)
    model = model.to_variant_model()
    cp_solver_1 = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
    cp_solver_1.init_model(output_type=True,
                           ignore_sec_objective=True,
                           add_objective_makespan=True)
    cp_solver_1.instance["maxt"] = 150
    cp_solver_1.instance["full_output"] = True
    cp_solver_1.instance.add_string("my_search=priority_smallest;\n")
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 35
    parameters_cp.TimeLimit_iter0 = 25
    parameters_cp.multiprocess = False
    parameters_cp.nb_process = 4
    results = cp_solver_1.solve(parameters_cp=parameters_cp)
    status = cp_solver_1.get_status_solver()
    if status == StatusSolver.OPTIMAL:
        print("Optimal solution..")
        return results

    from discrete_optimization.generic_tools.lns_mip import TrivialInitialSolution
    if len(results.list_solution_fits) > 0:
        initial_solution_provider = TrivialInitialSolution(results)
    else:
        initial_solution_provider = None

    cp_solver = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
    cp_solver.init_model(output_type=True,
                         ignore_sec_objective=False,
                         add_objective_makespan=False)
    cp_solver.instance["maxt"] = 150
    cp_solver.instance["full_output"] = True
    cp_solver.instance.add_string("my_search=priority_smallest;\n")
    constraint_handler = mix_lot(model,
                                 nb_cut_parts=[3, 4, 5, 6], #[3, 4, 5, 6], #[5, 6, 7],
                                 fraction_subproblems=[0.2, 0.3, 0.4, 0.5], #[0.1, 0.2, 0.3, 0.4, 0.5], #list(np.arange(0.05, 0.2, 0.15)),
                                 params_list=
                                 [ParamsConstraintBuilder(minus_delta_primary=80,
                                                          plus_delta_primary=80,
                                                          minus_delta_secondary=1,
                                                          plus_delta_secondary=1,
                                                          constraint_max_time_to_current_solution=True,
                                                          fraction_of_task_assigned_multiskill=0.45,
                                                          except_assigned_multiskill_primary_set=False,
                                                          first_method_multiskill=True,
                                                          second_method_multiskill=False,
                                                          additional_methods=False)],
                                 objective_subproblem=ObjectiveSubproblem.SUM_END_SUBTASKS,
                                 time_windows=False,
                                 equilibrate_multiskill_v2=False,
                                 time_window_length=10)
    lns = LargeNeighborhoodSearchScheduling(rcpsp_problem=model,
                                            cp_solver=cp_solver,
                                            constraint_handler=constraint_handler,
                                            partial_solution=None,
                                            post_process_solution=PostProMSRCPSPPreemptive(problem=model),
                                            initial_solution_provider=initial_solution_provider)
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 35
    parameters_cp.TimeLimit_iter0 = 25
    parameters_cp.multiprocess = False
    parameters_cp.nb_process = 4
    parameters_cp.free_search = False
    result_store = lns.solve(parameters_cp=parameters_cp,
                             nb_iteration_lns=1000,
                             nb_iteration_no_improvement=500,
                             max_time_seconds=time_limit,
                             skip_first_iteration=False,
                             stop_first_iteration_if_optimal=False)
    return result_store


def run_lns_cpopt(file=None, time_limit=1000, solver_lns: CPSolverName = CPSolverName.ORTOOLS):
    model = parse_dzn_file(file_path=file)
    model = model.to_variant_model()
    initial_solution_provider = None
    if initial_solution_provider is None:
        cp_solver_1 = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
        cp_solver_1.init_model(output_type=True,
                               model_type="mspsp",
                               ignore_sec_objective=True,
                               add_objective_makespan=True)
        cp_solver_1.instance["maxt"] = 1000
        cp_solver_1.instance["full_output"] = True
        cp_solver_1.instance.add_string("my_search=priority_smallest;\n")
        parameters_cp = ParametersCP.default()
        parameters_cp.TimeLimit = 35
        parameters_cp.TimeLimit_iter0 = 25
        parameters_cp.multiprocess = False
        parameters_cp.nb_process = 4
        parameters_cp.free_search = False
        results = cp_solver_1.solve(parameters_cp=parameters_cp)
        status = cp_solver_1.get_status_solver()
        if status == StatusSolver.OPTIMAL:
            print("Optimal solution..")
            return results
        if len(results.list_solution_fits) > 0:
            initial_solution_provider = TrivialInitialSolution(results)
        else:
            initial_solution_provider = None
    cp_solver = CP_MSPSP_MZN(model, cp_solver_name=solver_lns)
    cp_solver.init_model(output_type=True,
                         model_type="mspsp_compatible",
                         ignore_sec_objective=False,
                         add_objective_makespan=False)
    if initial_solution_provider is not None:
        sol, f = initial_solution_provider.get_starting_solution().get_best_solution_fit()
        eval_sol = model.evaluate(sol)
        cp_solver.instance["maxt"] = eval_sol["makespan"]+10
    else:
        cp_solver.instance["maxt"] = 170
    cp_solver.instance["full_output"] = True
    cp_solver.instance.add_string("my_search=assign_Then_start;\n")
    constraint_handler = mix_lot(model,
                                 nb_cut_parts=[2, 3, 4],
                                 # [3, 4, 5, 6], #[5, 6, 7],
                                 fraction_subproblems=[0.2, 0.3, 0.4, 0.5],
                                 # [0.1, 0.2, 0.3, 0.4, 0.5],
                                 # #list(np.arange(0.05, 0.2, 0.15)),
                                 params_list=
                                 [ParamsConstraintBuilder(minus_delta_primary=80,
                                                          plus_delta_primary=80,
                                                          minus_delta_secondary=1,
                                                          plus_delta_secondary=1,
                                                          constraint_max_time_to_current_solution=True,
                                                          fraction_of_task_assigned_multiskill=0.1,
                                                          except_assigned_multiskill_primary_set=False,
                                                          first_method_multiskill=True,
                                                          second_method_multiskill=False,
                                                          additional_methods=False)],
                                 objective_subproblem=ObjectiveSubproblem.SUM_END_SUBTASKS,
                                 time_windows=False,
                                 equilibrate_multiskill_v2=False,
                                 time_window_length=10)
    lns = LargeNeighborhoodSearchScheduling(rcpsp_problem=model,
                                            cp_solver=cp_solver,
                                            constraint_handler=constraint_handler,
                                            partial_solution=None,
                                            post_process_solution=PostProMSRCPSPPreemptive(problem=model),
                                            initial_solution_provider=initial_solution_provider)
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 35
    parameters_cp.TimeLimit_iter0 = 25
    parameters_cp.multiprocess = solver_lns in {CPSolverName.ORTOOLS, CPSolverName.CPOPT}
    parameters_cp.nb_process = 4
    parameters_cp.free_search = False
    result_store = lns.solve(parameters_cp=parameters_cp,
                             nb_iteration_lns=1000,
                             nb_iteration_no_improvement=500,
                             max_time_seconds=time_limit,
                             skip_first_iteration=True if initial_solution_provider is None else None,
                             stop_first_iteration_if_optimal=False)
    return result_store


def run_only_cp(file=None,
                time_limit=1000):
    model = parse_dzn_file(file_path=file)
    model = model.to_variant_model()
    cp_solver_1 = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
    cp_solver_1.init_model(output_type=True,
                           model_type="mspsp",
                           ignore_sec_objective=True,
                           add_objective_makespan=True)
    cp_solver_1.instance["maxt"] = 1000
    cp_solver_1.instance["full_output"] = True
    cp_solver_1.instance.add_string("my_search=priority_smallest;\n")
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = time_limit
    parameters_cp.TimeLimit_iter0 = 25
    parameters_cp.multiprocess = False
    parameters_cp.nb_process = 4
    parameters_cp.free_search = False
    results = cp_solver_1.solve(parameters_cp=parameters_cp)
    status = cp_solver_1.get_status_solver()
    if status == StatusSolver.OPTIMAL:
        print("Found optimal solution")
    return results, status


if __name__ == "__main__":
    # CP expe
    run_all_benchmark(cpopt=True,
                      name_folder="ortools-only-cp/",
                      solver=CPSolverName.ORTOOLS,
                      only_cp=True)
    # LNS-CP expe
    run_all_benchmark(cpopt=True,
                      name_folder="ortools-lns/",
                      solver=CPSolverName.ORTOOLS,
                      only_cp=False)




