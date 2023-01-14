from typing import List, Union

from discrete_optimization.rcpsp_multiskill.plots.plot_solution import MS_RCPSPModel, MS_RCPSPSolution, \
    MS_RCPSPSolution_Preemptive, compute_schedule_per_resource_individual_preemptive, \
    compute_schedule_per_resource_individual, \
    plt, Polygon, pp, PatchCollection
import numpy as np
from matplotlib.font_manager import FontProperties

from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import create_fake_tasks_multiskills


def compute_resource_consumption_ms(rcpsp_model: MS_RCPSPModel,
                                    rcpsp_sol: MS_RCPSPSolution,
                                    list_resources: List[Union[int, str]]=None,
                                    future_view=True):
    modes_dict = rcpsp_sol.modes
    last_activity = rcpsp_model.sink_task
    makespan = rcpsp_sol.get_end_time(last_activity)
    if list_resources is None:
        list_resources = rcpsp_model.resources_list
    consumptions = np.zeros((len(list_resources), makespan + 1))
    for act_id in rcpsp_model.tasks_list:
        for ir in range(len(list_resources)):
            use_ir = rcpsp_model.mode_details[act_id][modes_dict[act_id]].get(list_resources[ir], 0)
            sts = rcpsp_sol.get_start_times_list(act_id)
            ends = rcpsp_sol.get_end_times_list(act_id)
            if use_ir > 0:
                if not rcpsp_model.partial_preemption_data[act_id][modes_dict[act_id]].get(list_resources[ir], True):
                    st = sts[0]
                    end = ends[-1]
                    if future_view:
                        consumptions[ir, st + 1:end + 1] += use_ir
                    else:
                        consumptions[ir, st:end] += use_ir
                else:
                    for st, end in zip(sts, ends):
                        if future_view:
                            consumptions[ir, st + 1:end + 1] += use_ir
                        else:
                            consumptions[ir, st:end] += use_ir
    return consumptions, np.arange(0, makespan+1, 1)


def compute_nice_resource_consumption_ms(rcpsp_model: MS_RCPSPModel, rcpsp_sol: MS_RCPSPSolution,
                                         list_resources: List[Union[int, str]] = None):
    if list_resources is None:
        list_resources = rcpsp_model.resources_list
    c_future, times = compute_resource_consumption_ms(rcpsp_model, rcpsp_sol,
                                                      list_resources=list_resources,
                                                      future_view=True)
    c_past, times = compute_resource_consumption_ms(rcpsp_model, rcpsp_sol,
                                                    list_resources=list_resources,
                                                     future_view=False)
    merged_times = {i: [] for i in range(len(list_resources))}
    merged_cons = {i: [] for i in range(len(list_resources))}
    for r in range(len(list_resources)):
        for index_t in range(len(times)):
            merged_times[r] += [times[index_t], times[index_t]]
            merged_cons[r] += [c_future[r, index_t], c_past[r, index_t]]
    for r in merged_times:
        merged_times[r] = np.array(merged_times[r])
        merged_cons[r] = np.array(merged_cons[r])
    return merged_times, merged_cons


def compute_schedule_per_resource_individual_ms(rcpsp_model: MS_RCPSPModel,
                                                rcpsp_sol: MS_RCPSPSolution_Preemptive,
                                                resource_types_to_consider: List[str]=None,
                                                verbose=False):
    nb_ressources = len(rcpsp_model.resources_list)
    modes_dict = rcpsp_sol.modes
    if resource_types_to_consider is None:
        resources = rcpsp_model.resources_list
    else:
        resources = resource_types_to_consider
    sorted_task_by_start = sorted(rcpsp_sol.schedule,
                                  key=lambda x: 100000*rcpsp_sol.get_start_time(x)+rcpsp_model.index_task[x])
    sorted_task_by_end = sorted(rcpsp_sol.schedule,
                                key=lambda x: 100000*rcpsp_sol.get_end_time(x)+rcpsp_model.index_task[x])
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_start[0])
    print("Min time ", min_time)
    print("Max time ", max_time)
    with_calendar = rcpsp_model.is_varying_resource()

    array_ressource_usage = {resources[i]:
                            {"activity":
                             np.zeros((max_time-min_time+1,
                                       int(rcpsp_model.get_max_resource_capacity(resources[i])))),
                             "binary_activity":
                             np.zeros((max_time - min_time + 1,
                                       int(rcpsp_model.get_max_resource_capacity(resources[i])))),
                             "total_activity":
                             np.zeros(int(rcpsp_model.get_max_resource_capacity(resources[i]))),
                             "activity_last_n_hours":
                             np.zeros((max_time-min_time+1,
                                       int(rcpsp_model.get_max_resource_capacity(resources[i])))),
                             "boxes_time": []
                             }
                             for i in range(len(resources))}
    total_time = max_time-min_time+1
    nhour = int(min(8, total_time/2-1))
    index_to_time = {i: min_time+i for i in range(max_time-min_time+1)}
    time_to_index = {index_to_time[i]: i for i in index_to_time}

    for activity in sorted_task_by_start:
        mode = modes_dict[activity]
        start_times = rcpsp_sol.get_start_times_list(activity) #rcpsp_schedule[activity]["starts"]
        end_times = rcpsp_sol.get_end_times_list(activity) #[activity]["ends"]
        resources_needed = {r: rcpsp_model.mode_details[activity][mode].get(r, 0)
                            for r in resources}
        for r in resources_needed:
            if resources_needed[r] == 0:
                continue
            for start_time, end_time in zip(start_times, end_times):
                if end_time == start_time:
                    continue
                resources_needed = {r: rcpsp_model.mode_details[activity][mode].get(r, 0)
                                    for r in resources}
                stop_direct = False
                if not rcpsp_model.partial_preemption_data[activity][mode].get(r, True):
                    print("ac : ", activity, "mode : ", mode)
                    stop_direct = True
                    start_time = start_times[0]
                    end_time = end_times[-1]
                if r not in array_ressource_usage:
                    continue
                rneeded = resources_needed[r]
                range_interest = range(array_ressource_usage[r]["activity"].shape[1])
                while rneeded > 0:
                    # availables_people_r = [i for i in range(array_ressource_usage[r]["activity"].shape[1])
                    #                        if array_ressource_usage[r]["activity"][time_to_index[start_time], i] == 0]
                    availables_people_r = [i for i in range_interest
                                           if array_ressource_usage[r]["activity"][time_to_index[start_time], i] == 0]
                    if verbose:
                        print(len(availables_people_r), " people available : ")
                    if len(availables_people_r) > 0:
                        resource = min(availables_people_r,
                                       key=lambda x: array_ressource_usage[r]["total_activity"][x])
                        # greedy choice,
                        # the one who worked the less until now.
                        array_ressource_usage[r]["activity"][time_to_index[start_time]:time_to_index[end_time], resource] \
                            = rcpsp_model.index_task[activity] if isinstance(activity, str) else activity
                        array_ressource_usage[r]["binary_activity"][time_to_index[start_time]:time_to_index[end_time],
                                                                    resource] \
                            = 1
                        array_ressource_usage[r]["total_activity"][resource] += (end_time-start_time)
                        array_ressource_usage[r]["activity_last_n_hours"][:, resource] = \
                            np.convolve(array_ressource_usage[r]["binary_activity"][:, resource],
                                        np.array([1]*nhour+[0]+[0]*nhour),
                                        mode="same")
                        array_ressource_usage[r]["boxes_time"] += [[(resource-0.45, start_time+0.01,
                                                                     activity),
                                                                    (resource-0.45, end_time-0.01,
                                                                     activity),
                                                                    (resource+0.45, end_time-0.01,
                                                                     activity),
                                                                    (resource+0.45, start_time+0.01,
                                                                     activity),
                                                                    (resource-0.45, start_time+0.01,
                                                                     activity)]]
                        # for plot purposes.
                        rneeded -= 1
                    else:
                        print("r_needed ", rneeded)
                        print("Ressource needed : ", resources_needed)
                        print("ressource : ", r)
                        print("activity : ", activity)
                        print("Problem, can't build schedule")
                        print(array_ressource_usage[r]["activity"])
                        rneeded = 0
                if stop_direct:
                    break
    return array_ressource_usage


def plot_resource_individual_gantt_preemptive_with_ressource(rcpsp_model: MS_RCPSPModel,
                                                             rcpsp_sol: MS_RCPSPSolution_Preemptive,
                                                             title_figure="",
                                                             name_task=None,
                                                             subtasks=None,
                                                             fig=None,
                                                             ax=None,
                                                             current_t=None,
                                                             annotate_all_subparts=False,
                                                             one_color_per_task=False,
                                                             use_grid=True):
    array_employee_usage = compute_schedule_per_resource_individual_preemptive(rcpsp_model,
                                                                               rcpsp_sol)
    array_ressource_usage = compute_schedule_per_resource_individual_ms(rcpsp_model,
                                                                        rcpsp_sol)
    merged_times, merged_cons = compute_nice_resource_consumption_ms(rcpsp_model, rcpsp_sol,
                                                                     list_resources=rcpsp_model.resources_list)
    fake_tasks_res, fake_tasks_unit = create_fake_tasks_multiskills(rcpsp_model)
    sorted_task_by_start = sorted(rcpsp_sol.schedule,
                                  key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x])
    sorted_task_by_end = sorted(rcpsp_sol.schedule,
                                key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x])
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    sorted_employees = rcpsp_model.employees_list

    if name_task is None:
        name_task = {}
        for t in rcpsp_model.mode_details:
            name_task[t] = str(t)
    if subtasks is None:
        subtasks = set(rcpsp_model.tasks_list)
    # fig, ax = plt.subplots(len(array_ressource_usage),
    #                        figsize=(10, 5))
    # for i in range(len(array_ressource_usage)):
    #     ax[i].imshow(array_ressource_usage[resources_list[i]]["binary_activity"].T)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1+len(array_ressource_usage),
                               figsize=(12, 10))
        if title_figure != "":
            fig.suptitle(title_figure)
    position_label = {}
    #ax[0].set_title("Operators and ressources schedule")
    for i in range(len(sorted_employees)):
        patches = []
        nb_colors = len(sorted_task_by_start) // 2 if not one_color_per_task else len(sorted_task_by_start)
        # nb_colors = len(sorted_task_by_start)//2
        colors = plt.cm.get_cmap("hsv", nb_colors)
        for boxe in array_employee_usage[sorted_employees[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            if activity not in subtasks:
                continue
            x, y = polygon.exterior.xy
            ax[0].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=polygon.exterior.coords,
                              facecolor=colors((rcpsp_model.index_task[activity]) % nb_colors)))
            activity = boxe[0][2]
            if abs(boxe[0][1] - boxe[1][1]) >= 0.4:
                center = (sum([b[1] for b in boxe[:4]]) / 4 - 0.4, sum(b[0] for b in boxe[:4]) / 4)
                if activity not in position_label:
                    position_label[activity] = center
                position_label[activity] = max(center, position_label[activity])
                if annotate_all_subparts:
                    ax[0].annotate(name_task[activity],
                                   xy=center,
                                   # textcoords="offset points",
                                   font_properties=FontProperties(size=12,
                                                                  weight="bold"),
                                   verticalalignment='center',
                                   horizontalalignment="left",
                                   color="k",
                                   clip_on=True)
        current_employee = sorted_employees[i]
        unavailibilities_this_employee = [ft for ft in fake_tasks_unit if current_employee in ft]
        for j in range(len(unavailibilities_this_employee)):
            if unavailibilities_this_employee[j]["start"]>=max_time:
                continue
            print(unavailibilities_this_employee[j])
            resource = i
            start_time = unavailibilities_this_employee[j]["start"]
            end_time = start_time+unavailibilities_this_employee[j]["duration"]
            box = [(resource-0.45, min(max_time, start_time+0.01)),
                   (resource-0.45, min(max_time, end_time-0.01)),
                   (resource+0.45, min(max_time, end_time-0.01)),
                   (resource+0.45, min(max_time, start_time+0.01)),
                   (resource-0.45, min(max_time, start_time+0.01))]
            polygon = Polygon([(b[1], b[0]) for b in box])
            x, y = polygon.exterior.xy
            ax[0].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=polygon.exterior.coords,
                              facecolor="red", alpha=0.2, zorder=0))
            center = ((end_time+start_time-1)/2, sum(b[0] for b in box[:4]) / 4)
            ax[0].annotate("Break n°"+str(j+1),
                           xy=center,
                               # textcoords="offset points",
                           font_properties=FontProperties(size=12,
                                                          weight="bold"),
                           verticalalignment='center',
                           horizontalalignment="left",
                           color="k",
                           clip_on=True)

        p = PatchCollection(patches,
                            match_original=True,
                            # cmap=matplotlib.cm.get_cmap('Blues'),
                            alpha=0.4)
        ax[0].add_collection(p)
        ax[0].set_xlim((min_time, max_time))
        ax[0].set_ylim((-0.5, len(sorted_employees)-1+0.5))
        ax[0].set_yticks(range(len(sorted_employees)))
        ax[0].set_yticklabels(tuple(sorted_employees),
                              fontdict={"size": 10})
        for activity in position_label:
            ax[0].annotate(name_task[activity],
                           xy=position_label[activity],
                           # textcoords="offset points",
                           font_properties=FontProperties(size=12,
                                                          weight="bold"),
                           verticalalignment='center',
                           horizontalalignment="left",
                           color="k",
                           clip_on=True)
        if use_grid:
            ax[0].grid(True)
        if current_t is not None:
            ax[0].axvline(x=current_t, label='current time', color='r', ls='--')

    resources_list = rcpsp_model.resources_list
    x_lim = None
    for i in range(len(resources_list)):
        patches = []
        nb_colors = len(sorted_task_by_start) // 2 if not one_color_per_task else len(sorted_task_by_start)
        colors = plt.cm.get_cmap("hsv", nb_colors)
        for boxe in array_ressource_usage[resources_list[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            x, y = polygon.exterior.xy
            ax[i+1].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=polygon.exterior.coords,
                              facecolor=colors((rcpsp_model.index_task[activity]) % nb_colors)))
            if abs(boxe[0][1] - boxe[1][1]) >= 0.4:
                center = (sum([b[1] for b in boxe[:4]]) / 4 - 0.4, sum(b[0] for b in boxe[:4]) / 4)
                if activity not in position_label:
                    position_label[activity] = center
                position_label[activity] = max(center, position_label[activity])
                if annotate_all_subparts:
                    ax[i+1].annotate(name_task[activity],
                                     xy=center,
                                      # textcoords="offset points",
                                     font_properties=FontProperties(size=12,
                                                                    weight="bold"),
                                     verticalalignment='center',
                                     horizontalalignment="left",
                                     color="k",
                                     clip_on=True)
        #ax[i+1].plot(merged_times[i],
        #             [rcpsp_model.get_resource_available(resources_list[i], m)-0.4
        #              for m in merged_times[i]], linestyle="--",
        #             label="Limit : " + str(resources_list[i]), zorder=0)
        p = PatchCollection(patches,
                            match_original=True,
                            # cmap=matplotlib.cm.get_cmap('Blues'),
                            alpha=0.4)
        ax[i+1].add_collection(p)
        # ax[i+1].set_title("Ressource "+resources_list[i])
        if x_lim is None:
            ax[i+1].set_xlim((min_time, max_time))
        else:
            ax[i+1].set_xlim(x_lim)
        ax[i+1].set_ylim((-0.5, rcpsp_model.get_max_resource_capacity(resources_list[i])-1+0.5))
        ax[i+1].set_yticks(range(int(rcpsp_model.get_max_resource_capacity(resources_list[i]))))
        ax[i+1].set_yticklabels(tuple([resources_list[i]+"-"+str(j) for j in
                                       range(int(rcpsp_model.get_max_resource_capacity(resources_list[i])))]),
                                fontdict={"size": 10})
        if use_grid:
            ax[i+1].grid(True)
        if current_t is not None:
            ax[i+1].axvline(x=current_t, label='current time', color='r', ls='--')
    return fig

