from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, \
     Employee, SkillDetail, SpecialConstraintsDescription


def parse_file(path="A1.dat"):
    file = path
    with open(file, "r") as f:
        input_data = f.read()
        lines = input_data.split("\n")
        first_line = lines[0].split()
        horizon = int(first_line[0])
        nb_ressource = int(first_line[1])
        nb_skills = int(first_line[2])
        nb_tasks = int(first_line[3])
        nb_worker = int(first_line[4])
        ressource_availability = {i: [] for i in range(nb_ressource)}
        worker_availability = {j: [] for j in range(nb_worker)}
        skills_per_worker = {j: None for j in range(nb_worker)}
        line_ressource = lines[1]
        ressource_capa = line_ressource.split()
        for i in range(nb_ressource):
            ressource_availability[i] = [int(ressource_capa[i])]*horizon
        lines_workers = lines[2:2+nb_worker]
        ind = 2+nb_worker
        for j in range(nb_worker):
            worker_availability[j] = [int(k) for k in lines_workers[j].split()]
        lines_skills = lines[ind+1:ind+1+nb_worker]
        for j in range(nb_worker):
            skills_per_worker[j] = [int(k) for k in lines_skills[j].split()]
        ind = ind+1+nb_worker+1
        durations_tasks = [int(x) for x in lines[ind].split()]
        precedence = {i: [] for i in range(nb_tasks)}
        successors = {i: [] for i in range(nb_tasks)}
        lines_precedence = lines[ind+1:ind+nb_tasks+1]
        for k in range(nb_tasks):
            l = lines_precedence[k].split()
            precedence[k] = [int(jj) for jj in range(len(l))
                             if int(l[jj]) == 1]
            for pp in precedence[k]:
                successors[pp] += [k]
        ind = ind+nb_tasks+2
        line_preemptive_tech = lines[ind].split()
        preemptive_tech = [not int(x) == 1 for x in line_preemptive_tech]
        ind = ind+1
        lines_preemptive_ressource = lines[ind:ind+nb_tasks]
        preemptive_ressource = {j: {r: True for r in range(nb_ressource)}
                                for j in range(nb_tasks)}
        for j in range(len(lines_preemptive_ressource)):
            l = lines_preemptive_ressource[j].split()
            for k in range(len(l)):
                preemptive_ressource[j][k] = not (int(l[k])==1)
        ind = ind+nb_tasks+1
        lines_skills_requirements = lines[ind:ind+nb_tasks]
        skills_requirement = {j: {s: 0 for s in range(nb_skills)}
                              for j in range(nb_tasks)}
        for k in range(len(lines_skills_requirements)):
            l = lines_skills_requirements[k].split()
            for jj in range(len(l)):
                skills_requirement[k][jj] = int(l[jj])

        ind = ind + nb_tasks + 1
        lines_ressource_requirements = lines[ind:ind + nb_tasks]
        ressource_requirement = {j: {s: 0 for s in range(nb_skills)}
                                 for j in range(nb_tasks)}
        for k in range(len(lines_ressource_requirements)):
            l = lines_ressource_requirements[k].split()
            for jj in range(len(l)):
                ressource_requirement[k][jj] = int(l[jj])

        line_deadline = lines[ind+nb_tasks+1]
        deadlines = [int(x)-1 for x in line_deadline.split()]
        print(deadlines)
        release = [int(x)-1 for x in lines[ind+nb_tasks+2].split()]
        number_of_worker = [int(x) for x in lines[ind+nb_tasks+3].split()]
        deadline_type = [True if int(x) == 1 else False for x in lines[ind+nb_tasks+3].split()]
        return horizon, nb_tasks, nb_ressource, nb_skills, nb_worker,ressource_availability, \
               worker_availability, skills_per_worker, skills_requirement, ressource_requirement, \
               ressource_capa,precedence, successors,preemptive_tech, preemptive_ressource, \
               deadlines, release, deadline_type, durations_tasks, number_of_worker


def parse_file_to_model(path="A1.dat", double_horizon=False):
    horizon, nb_tasks, nb_ressource, nb_skills, nb_worker, ressource_availability, \
    worker_availability, skills_per_worker, skills_requirement, ressource_requirement, \
    ressource_capa, precedence, successors, preemptive_tech, preemptive_ressource, \
    deadlines, release, deadline_type, durations_tasks, number_of_worker = parse_file(path)

    tasks_list = ["source"] + [i for i in range(nb_tasks)] + ["sink"]
    mode_details = {t: {1: {"R"+str(r): ressource_requirement[t][r]
                            for r in ressource_requirement[t]}}
                    for t in ressource_requirement}
    for t in ressource_requirement:
        for s in range(nb_skills):
            mode_details[t][1]["S"+str(s)] = skills_requirement[t][s]
        mode_details[t][1]["W"] = number_of_worker[t]

    for t in mode_details:
        mode_details[t][1]["duration"] = durations_tasks[t]
    for node in ["source", "sink"]:
        mode_details[node] = {1: {"R"+str(r): 0 for r in range(nb_ressource)}}
        for s in range(nb_skills):
            mode_details[node][1]["S"+str(s)] = 0
        mode_details[node][1]["W"] = 0
        mode_details[node][1]["duration"] = 0
    successors["source"] = [i for i in range(nb_tasks)]+["sink"]
    successors["sink"] = []
    for i in range(nb_tasks):
        successors[i] += ["sink"]
    ressource = {"R"+str(i): ressource_availability[i]*2 if double_horizon else ressource_availability[i]
                 for i in range(nb_ressource)}
    workers_dict = {emp: Employee(dict_skill={"S"+str(k):
                                              SkillDetail(skills_per_worker[emp][k], 1, 1)
                                              for k in range(len(skills_per_worker[emp]))
                                              if skills_per_worker[emp][k] > 0},
                                  calendar_employee=worker_availability[emp]*2
                                  if double_horizon else worker_availability[emp])
                    for emp in range(nb_worker)}
    for emp in workers_dict:
        workers_dict[emp].dict_skill["W"] = SkillDetail(1, 1, 1)
    preemptive_indicator = {i: preemptive_tech[i] for i in range(nb_tasks)}
    preemptive_indicator["source"] = False
    preemptive_indicator["sink"] = False
    partial_preemption_data = {t: {1: {"R"+str(i): preemptive_ressource[t][i]
                                       for i in range(len(preemptive_ressource[t]))}}
                               for t in preemptive_ressource}
    partial_preemption_data["source"] = {1: {"R"+str(i): True for i in range(nb_ressource)}}
    partial_preemption_data["sink"] = {1: {"R"+str(i): True for i in range(nb_ressource)}}
    special_constraint = SpecialConstraintsDescription(start_times_window={t: (release[t], None)
                                                                           for t in range(len(release))},
                                                       end_times_window={t: (None, deadlines[t])
                                                                         for t in range(len(deadlines))
                                                                         if deadline_type[t]})
    #print(horizon)
    model = MS_RCPSPModel(skills_set={"S"+str(k) for k in range(nb_skills)}.union({"W"}),
                          resources_set={"R"+str(k) for k in range(nb_ressource)},
                          non_renewable_resources=set(),
                          resources_availability=ressource,
                          employees=workers_dict,
                          employees_availability=[],
                          mode_details=mode_details,
                          successors=successors,
                          horizon=2*horizon if double_horizon else horizon,
                          tasks_list=tasks_list,
                          employees_list=[i for i in range(nb_worker)],
                          horizon_multiplier=1,
                          sink_task="sink",
                          source_task="source",
                          preemptive=True,
                          preemptive_indicator=preemptive_indicator,
                          special_constraints=special_constraint,
                          partial_preemption_data=partial_preemption_data,
                          strictly_disjunctive_subtasks=False)
    model = model.to_variant_model()
    return model









