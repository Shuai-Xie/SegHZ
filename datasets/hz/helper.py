def reorder_idxs(cdict):
    idxs = sorted(set(cdict.values()))  # 合并后剩下的类别 idx
    new_map = {val: i + 1 for i, val in enumerate(idxs)}  # 将零散的 val 重新接续编排
    for k in cdict:
        cdict[k] = new_map[cdict[k]]  # val -> new idx


def get_merge_map_idxs(merge_all_buildings=False):
    OpenAreas = [4, 5]  # SubUrban_OpenArea + Urban_OpenArea

    High_Buildings = [9, 10, 11, 12, 13, 14, 15]
    Paralle_Regular_Buildings = [16]
    Irregular_Buildings = [17, 18]

    Green_land = [6, 19]  # Green_land + SubUrban_Village

    if merge_all_buildings:
        High_Buildings += (Paralle_Regular_Buildings + Irregular_Buildings)

    merge_list = [OpenAreas, High_Buildings, Green_land]

    old2new = {i: i for i in range(1, 21)}  # {old:new}

    # 每个合并子区间，使用首个 idx 代表父类
    for idxs in merge_list:
        for i in idxs:
            old2new[i] = idxs[0]

    # 去掉不存在的 [2, 3]
    for i in [2, 3]:
        old2new.pop(i)

    # 重新编排，得到连续 cls idx
    reorder_idxs(old2new)
    return old2new


def get_merge_func(merge_all_buildings=False):
    merge_map_idxs = get_merge_map_idxs(merge_all_buildings)
    max_idx = max(merge_map_idxs.values())

    def merge_func(target):
        for old_i, new_i in merge_map_idxs.items():
            if old_i != new_i:
                target[target == old_i] = new_i
        target[target > max_idx] = 0  # bg=0
        return target

    return merge_func


if __name__ == '__main__':
    print(get_merge_map_idxs())
    print(get_merge_map_idxs(merge_all_buildings=True))
