import json
import os
import random
import time
from typing import Tuple

from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import ParamsConstraintBuilder, ObjectiveSubproblem
from discrete_optimization.generic_rcpsp_tools.solution_repair import NeighborRepairProblems
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.lns_mip import InitialSolutionFromSolver, TrivialInitialSolution
from discrete_optimization.generic_tools.result_storage.result_storage import merge_results_storage, ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.solver.ls_solver import LS_RCPSP_Solver, LS_SOLVER
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import plot_resource_individual_gantt_preemptive
from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import plot_ressource_view, plt
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, MS_RCPSPModel_Variant, \
    MS_RCPSPSolution_Variant, Employee, SkillDetail, SpecialConstraintsDescription, MS_RCPSPSolution_Preemptive_Variant, \
    MS_RCPSPSolution, compute_ressource_array_preemptive
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE, \
    CPSolverName, SearchStrategyMS_MRCPSP
from discrete_optimization.generic_rcpsp_tools.neighbor_builder import mix_lot
from discrete_optimization.rcpsp_multiskill.solvers.lns_post_process_rcpsp import PostProMSRCPSPPreemptive
from scripts_paper.experiments_ppremmption.parser_mspsp_pp import parse_file_to_model
import logging
logging.basicConfig(level=logging.INFO)
this_folder = os.path.dirname(os.path.abspath(__file__))
folder_data = os.path.join(this_folder, "data_pub/")
files = [os.path.join(folder_data, f) for f in os.listdir(folder_data) if ".dat" in f]
folder_results = os.path.join(this_folder, "results/")
if not os.path.exists(folder_results):
    os.makedirs(folder_results)


def run_solver(path=os.path.join(folder_data, "A1.dat"),
               nb_iteration_lns=2000,
               max_time_seconds=1500,
               double_horizon: bool = False):
    model = parse_file_to_model(path=path, double_horizon=double_horizon)
    special_constraint = model.special_constraints
    solution = model.get_dummy_solution(preemptive=True)
    print("Nb task preempted :", len(solution.get_task_preempted()))
    print('max preempted',  solution.get_max_preempted())
    print('nb preempted',  solution.get_nb_task_preemption())
    print(model.evaluate(solution))
    from discrete_optimization.generic_rcpsp_tools.large_neighborhood_search_scheduling import\
        LargeNeighborhoodSearchScheduling
    from discrete_optimization.generic_tools.result_storage.result_storage import from_solutions_to_result_storage
    init_permutation = sorted(list(special_constraint.end_times_window),
                              key=lambda x: special_constraint.end_times_window[x][1])
    indexes = [model.index_task_non_dummy[t] for t in init_permutation]
    indexes += [model.index_task_non_dummy[t]
                for t in model.tasks_list_non_dummy if model.index_task_non_dummy[t] not in indexes]
    init_solution = MS_RCPSPSolution_Preemptive_Variant(problem=model,
                                                        priority_list_task=indexes,
                                                        modes_vector=[1 for i in range(model.n_jobs_non_dummy)],
                                                        priority_worker_per_task=[[w for w in model.employees]
                                                                                  for i in range(model.n_jobs_non_dummy)],
                                                        fast=True)

    init = InitialSolutionFromSolver(
        LS_RCPSP_Solver(model=model,
                        ls_solver=LS_SOLVER.SA),
        nb_iteration_max=10000,
        # temperature=1000,
        # decay_temperature=0.99999999999,
        init_solution_process=False,
        starting_point=init_solution)
    res = init.get_starting_solution()

    init = TrivialInitialSolution(solution=res)
    best_solution: MS_RCPSPSolution_Preemptive_Variant = res.get_best_solution()
    print("satisfy ", model.satisfy(best_solution))
    max_nb_preemption = best_solution.get_max_preempted()
    max_preempted_task = best_solution.get_nb_task_preemption()
    cp_solver = CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE(rcpsp_model=model,
                                                    cp_solver_name=CPSolverName.CPOPT)
    cp_solver.init_model(max_time=model.evaluate(best_solution)["makespan"]+10,
                         max_preempted=max_preempted_task+3,
                         nb_preemptive=max_nb_preemption+1,
                         possibly_preemptive=[model.preemptive_indicator.get(t, True)
                                              for t in model.tasks_list],
                         partial_solution=model.special_constraints,
                         ignore_sec_objective=False,
                         add_objective_makespan=False,
                         output_type=True,
                         add_partial_solution_hard_constraint=False,
                         unit_usage_preemptive=True)
    post_pro = None
    p = [ParamsConstraintBuilder(minus_delta_primary=200,
                                 plus_delta_primary=200,
                                 minus_delta_secondary=5,
                                 plus_delta_secondary=5,
                                 minus_delta_primary_duration=5,
                                 plus_delta_primary_duration=5,
                                 minus_delta_secondary_duration=2,
                                 plus_delta_secondary_duration=2,
                                 constraint_max_time_to_current_solution=True,
                                 fraction_of_task_assigned_multiskill=0.98),
         ParamsConstraintBuilder(minus_delta_primary=200,
                                 plus_delta_primary=200,
                                 minus_delta_secondary=5,
                                 plus_delta_secondary=5,
                                 minus_delta_primary_duration=10,
                                 plus_delta_primary_duration=10,
                                 minus_delta_secondary_duration=1,
                                 plus_delta_secondary_duration=1,
                                 constraint_max_time_to_current_solution=False,
                                 fraction_of_task_assigned_multiskill=0.8)]
    if "(3)" in path:
        p = [ParamsConstraintBuilder(minus_delta_primary=200,
                                     plus_delta_primary=200,
                                     minus_delta_secondary=15,
                                     plus_delta_secondary=15,
                                     minus_delta_primary_duration=5,
                                     plus_delta_primary_duration=5,
                                     minus_delta_secondary_duration=2,
                                     plus_delta_secondary_duration=2,
                                     constraint_max_time_to_current_solution=True,
                                     fraction_of_task_assigned_multiskill=0.93),
         ParamsConstraintBuilder(minus_delta_primary=200,
                                 plus_delta_primary=200,
                                 minus_delta_secondary=10,
                                 plus_delta_secondary=10,
                                 minus_delta_primary_duration=10,
                                 plus_delta_primary_duration=10,
                                 minus_delta_secondary_duration=3,
                                 plus_delta_secondary_duration=3,
                                 constraint_max_time_to_current_solution=False,
                                 fraction_of_task_assigned_multiskill=0.75)]
    constraint_handler = mix_lot(model,
                                 nb_cut_parts=[1, 2, 3, 4],
                                 fraction_subproblems=[0.2,
                                                       0.3,
                                                       0.4, 0.5, 0.6],
                                 params_list=
                                 p,
                                 generalized_precedence_constraint=True,
                                 use_makespan_of_subtasks=False,
                                 equilibrate_multiskill=True)
    lns = LargeNeighborhoodSearchScheduling(rcpsp_problem=model,
                                            partial_solution=special_constraint,
                                            cp_solver=cp_solver,
                                            post_process_solution=post_pro,
                                            initial_solution_provider=init,
                                            constraint_handler=constraint_handler)
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 100
    parameters_cp.TimeLimit_iter0 = 100
    parameters_cp.free_search = True
    result_store = lns.solve(parameters_cp=parameters_cp,
                             nb_iteration_lns=nb_iteration_lns,
                             nb_iteration_no_improvement=1000,
                             max_time_seconds=max_time_seconds,
                             skip_first_iteration=False,
                             stop_first_iteration_if_optimal=True)
    best_solution = result_store.get_best_solution_fit()[0]
    do_post_process = False
    if do_post_process:
        constraint_handler = NeighborRepairProblems(problem=model,
                                                    params_list=
                                                    [ParamsConstraintBuilder(minus_delta_primary=200,
                                                                             plus_delta_primary=200,
                                                                             minus_delta_secondary=20,
                                                                             plus_delta_secondary=20,
                                                                             constraint_max_time_to_current_solution
                                                                             =False),
                                                     ParamsConstraintBuilder(minus_delta_primary=200,
                                                                             plus_delta_primary=200,
                                                                             minus_delta_secondary=2,
                                                                             plus_delta_secondary=2,
                                                                             constraint_max_time_to_current_solution
                                                                             =False)])
        lns = LargeNeighborhoodSearchScheduling(rcpsp_problem=model,
                                                partial_solution=special_constraint,
                                                cp_solver=cp_solver,
                                                initial_solution_provider=
                                                TrivialInitialSolution(from_solutions_to_result_storage([best_solution],
                                                                                                        model)),
                                                constraint_handler=constraint_handler)
        result_store_2 = lns.solve(parameters_cp=parameters_cp,
                                   nb_iteration_lns=5,
                                   nb_iteration_no_improvement=1000,
                                   max_time_seconds=1000,
                                   skip_first_iteration=False,
                                   stop_first_iteration_if_optimal=True)
        res = merge_results_storage(result_store, result_store_2)
        best_solution = res.get_best_solution_fit()[0]
    print(best_solution.get_task_preempted())
    print(model.evaluate(best_solution))
    return result_store, best_solution, model


def run_ls_solver(path="A1.dat",
                  max_time_seconds=1500,
                  double_horizon: bool = False):
    model = parse_file_to_model(path=path, double_horizon=double_horizon)
    special_constraint = model.special_constraints

    init_permutation = sorted(list(special_constraint.end_times_window),
                              key=lambda x: special_constraint.end_times_window[x][1])
    indexes = [model.index_task_non_dummy[t] for t in init_permutation]
    indexes += [model.index_task_non_dummy[t]
                for t in model.tasks_list_non_dummy if model.index_task_non_dummy[t] not in indexes]
    init_solution = MS_RCPSPSolution_Preemptive_Variant(problem=model,
                                                        priority_list_task=indexes,
                                                        modes_vector=[1 for i in range(model.n_jobs_non_dummy)],
                                                        priority_worker_per_task=[[w for w in model.employees]
                                                                                  for i in range(model.n_jobs_non_dummy)],
                                                        fast=True)
    t = time.time()
    init = InitialSolutionFromSolver(
        LS_RCPSP_Solver(model=model,
                        ls_solver=LS_SOLVER.SA),
        nb_iteration_max=30000,
        # temperature=1000,
        # decay_temperature=0.99999999999,
        init_solution_process=False,
        max_time_seconds=max_time_seconds,
        starting_point=init_solution)
    res = init.get_starting_solution()
    t_end = time.time()
    init = TrivialInitialSolution(solution=res)
    best_solution: MS_RCPSPSolution_Preemptive_Variant = res.get_best_solution()
    print("satisfy ", model.satisfy(best_solution))
    max_nb_preemption = best_solution.get_max_preempted()
    max_preempted_task = best_solution.get_nb_task_preemption()
    return res, best_solution, model


def run_cp():
    for path in files:
        print(path)
        model = parse_file_to_model(path=path, double_horizon=True)
        # for emp in model.employees:
        #    model.employees[emp].calendar_employee = [1]*len(model.employees[emp].calendar_employee)
        init_permutation = sorted(list(model.special_constraints.end_times_window),
                                  key=lambda x: model.special_constraints.end_times_window[x][1])
        indexes = [model.index_task_non_dummy[t] for t in init_permutation]
        indexes += [model.index_task_non_dummy[t]
                    for t in model.tasks_list_non_dummy if model.index_task_non_dummy[t] not in indexes]

        init_solution = MS_RCPSPSolution_Preemptive_Variant(problem=model,
                                                            priority_list_task=[2, 39, 13, 16, 11, 43, 31, 4, 26, 8, 1, 42,
                                                                                29, 40, 37, 21, 30, 23, 49, 22, 20, 6, 24,
                                                                                5, 44, 45, 36, 12, 27, 41, 46, 25, 34, 33,
                                                                                17, 7, 19, 14, 10, 28, 47, 0, 9, 48, 38,
                                                                                32, 18, 3, 35, 15],
                                                            modes_vector=[1 for i in range(model.n_jobs_non_dummy)],
                                                            priority_worker_per_task=[[w for w in model.employees]
                                                                                      for i in
                                                                                      range(model.n_jobs_non_dummy)],
                                                            fast=True)
        # init = InitialSolutionFromSolver(
        #     LS_RCPSP_Solver(model=model,
        #                     ls_solver=LS_SOLVER.SA),
        #     nb_iteration_max=2000,
        #     nb_iteration_no_improvement=50,
        #     # temperature=1000,
        #     # decay_temperature=0.99999999999,
        #     init_solution_process=False,
        #     starting_point=init_solution)
        # res = init.get_starting_solution()
        from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import compute_constraints_details
        #details = compute_constraints_details(res.get_best_solution_fit()[0], constraints=model.special_constraints)
        #print(details)
        cp_solver_init = CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE(rcpsp_model=model,
                                                             cp_solver_name=CPSolverName.CPOPT)
        p = ParametersCP.default()
        p.free_search = False
        p.TimeLimit = 1000
        cp_solver_init.init_model(max_time=500,
                                  max_preempted=2,
                                  nb_preemptive=3,
                                  possibly_preemptive=[model.preemptive_indicator.get(t, False)
                                                       for t in model.tasks_list],
                                  partial_solution=model.special_constraints,
                                  ignore_sec_objective=True,
                                  add_objective_makespan=True,
                                  output_type=True,
                                  fake_tasks=True,
                                  exact_skills_need=False,
                                  add_partial_solution_hard_constraint=False,
                                  unit_usage_preemptive=True)
        res = cp_solver_init.solve(p)
        sol = res.get_best_solution_fit()
        print(sol)


def run_all_benchmark():
    random.shuffle(files)
    folder_results_v4 = os.path.join(this_folder, "results_lns/")
    if not os.path.exists(folder_results_v4):
        os.makedirs(folder_results_v4)
    cur_fol = folder_results_v4
    for path in files:
        basepath = os.path.basename(path)
        if "best_solution_"+str(basepath)+".json" in os.listdir(cur_fol):
            print("Done", basepath)
            continue
        print("Doing ..", path)
        try:
            t = time.time()
            res, best_solution, model = run_solver(path=path,
                                                   max_time_seconds=400,
                                                   nb_iteration_lns=2000,
                                                   double_horizon=True)
            t_end = time.time()
            import pickle
            export = True
            debug = False
            if export:
                pickle.dump(best_solution, open(os.path.join(cur_fol, "best_solution_"+str(basepath)+".pk"), "wb"))
                pickle.dump(res, open(os.path.join(cur_fol, "result_store_"+str(basepath)+".pk"),
                                      "wb"))
                best_solution_fit: Tuple[MS_RCPSPSolution, float] = res.get_best_solution_fit()
                eval = model.evaluate(best_solution_fit[0])
                dict_ = {"file": path,
                         "schedule": {k: {p: [int(x) for x in best_solution_fit[0].schedule[k][p]]
                                          for p in best_solution_fit[0].schedule[k]}
                                      for k in best_solution_fit[0].schedule},
                         "employee_usage": {t: [{emp: list(best_solution_fit[0].employee_usage[t][i][emp])
                                                for emp in best_solution_fit[0].employee_usage[t][i]}
                                                for i in range(len(best_solution_fit[0].employee_usage[t]))]
                                            for t in best_solution_fit[0].employee_usage},
                         "runtime": int(t_end - t),
                         "eval": {p: int(eval[p]) for p in eval},
                         "makespan": int(best_solution_fit[0].problem.evaluate(best_solution_fit[0])["makespan"])}

                json.dump(dict_,
                          open(os.path.join(cur_fol, "best_solution_"+str(basepath)+".json"), "w"),
                          indent=4, default=list)
        except Exception as e:
            json.dump([str(e)], open("bug_"+str(basepath)+".json", "w"))


def run_all_benchmark_ls():
    random.shuffle(files)
    folder_results_ls = os.path.join(this_folder, "results_ls/")
    if not os.path.exists(folder_results_ls):
        os.makedirs(folder_results_ls)
    cur_fol = folder_results_ls
    for path in files:
        basepath = os.path.basename(path)
        if "best_solution_"+str(basepath)+".json" in os.listdir(folder_results_ls):
            print("Done", basepath)
            continue
        print("Doing ..", path)
        try:
            t = time.time()
            res, best_solution, model = run_ls_solver(path=path,
                                                      max_time_seconds=1500,
                                                      double_horizon=False)
            t_end = time.time()
            import pickle
            export = True
            debug = False
            # ressource_array = compute_ressource_array_preemptive(model, best_solution)
            # for resource in ressource_array:
            #     print(res, min(ressource_array[resource]))
            if debug:
                ressource_array = compute_ressource_array_preemptive(model, best_solution)
                fig, ax = plt.subplots(1)
                for r in ressource_array:
                    print(r, min(ressource_array[r]))
                    ax.plot(ressource_array[r], label=r)
                ax.legend()
                plt.show()
            if export:
                pickle.dump(best_solution, open(os.path.join(cur_fol, "best_solution_"+str(basepath)+".pk"), "wb"))
                pickle.dump(res, open(os.path.join(cur_fol, "result_store_"+str(basepath)+".pk"),
                                      "wb"))
                best_solution_fit: Tuple[MS_RCPSPSolution, float] = res.get_best_solution_fit()
                eval = model.evaluate(best_solution_fit[0])
                dict_ = {"file": path,
                         "schedule": {k: {p: [int(x) for x in best_solution_fit[0].schedule[k][p]]
                                          for p in best_solution_fit[0].schedule[k]}
                                      for k in best_solution_fit[0].schedule},
                         "employee_usage": {t: [{emp: list(best_solution_fit[0].employee_usage[t][i][emp])
                                                for emp in best_solution_fit[0].employee_usage[t][i]}
                                                for i in range(len(best_solution_fit[0].employee_usage[t]))]
                                            for t in best_solution_fit[0].employee_usage},
                         "runtime": int(t_end - t),
                         "eval": {p: int(eval[p]) for p in eval},
                         "makespan": int(best_solution_fit[0].problem.evaluate(best_solution_fit[0])["makespan"])}

                json.dump(dict_,
                          open(os.path.join(cur_fol, "best_solution_"+str(basepath)+".json"), "w"),
                          indent=4, default=list)
        except Exception as e:
            json.dump([str(e)], open("bug_"+str(basepath)+".json", "w"))


if __name__ == "__main__":
    run_all_benchmark_ls()
    run_all_benchmark()
    run_cp()

