import os


def run(file_name):
    first_row, streets, paths = read_file(file_name + ".txt")
    sol = get_solution(first_row, streets, paths)
    with open(os.path.join("outputs", file_name + "_out.txt"), "w") as f_obj:
        f_obj.writelines([str(row) + "\n" for row in sol])


def read_file(file_name):
    with open(os.path.join("inputs", file_name), "r") as f_obj:
        rows = f_obj.read().split("\n")
    first_row = [int(i) for i in rows[0].split()]
    streets = []
    for row in rows[1:1 + first_row[2]]:
        r = row.split()
        streets.append((int(r[0]), int(r[1]), r[2], int(r[3])))

    paths = []
    for row in rows[1 + first_row[2]:-1]:
        r = row.split()
        r[0] = int(r[0])
        paths.append(tuple(r))

    assert len(paths) == first_row[3]
    return first_row, streets, paths


def filter_paths(paths, D, streets, factor=1.0):
    streets_t = {s[2]: s[3] for s in streets}
    return [p for p in paths if get_path_time(p, streets_t) <= D * factor]


def get_path_time(p, streets_t):
    return sum(streets_t[s] for s in p[1:])


def get_solution(first_row, streets, paths):
    D, I, S, V, F = first_row
    paths = filter_paths(paths, D, streets, factor=1.0)
    street_ends = {s[2]: s[1] for s in streets}
    # Street counts
    street_cnt = {s: 0 for s in street_ends.keys()}
    for p in paths:
        for _p in p[1:]:
            street_cnt[_p] += 1

    # Street counts beginning
    street_cnt_b = {s: 0 for s in street_ends.keys()}
    for p in paths:
        street_cnt_b[p[1]] += 1

    int_streets = {}
    for st, i in street_ends.items():
        if street_cnt[st]:
            if i not in int_streets:
                int_streets[i] = [st]
            else:
                int_streets[i].append(st)

    file_rows = [len(int_streets)]
    for i, _int_st in int_streets.items():
        if len(_int_st) == 1:
            schedules = ["%s 1" % _int_st[0]]
        else:
            counts = {n: street_cnt[n] for n in _int_st}
            b_counts = {n: street_cnt_b[n] for n in _int_st}
            if max(b_counts.values()) == 0:
                m = min(counts.values())
                schedules = ["%s %s" % (s, max(1, int(x*0.3 // m))) for s, x in counts.items()]
            else:
                _int_st = sorted(_int_st, key=lambda x: b_counts[x], reverse=True)
                m = min(counts.values())
                schedules = ["%s %s" % (s, max(1, int(counts[s]*0.3 // m))) for s in _int_st]

        file_rows += [i, len(_int_st), *schedules]

    return file_rows


if __name__ == "__main__":
    for f in ("a", "b", "c", "d", "e", "f"):
        run(f)
