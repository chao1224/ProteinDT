
task_list = ["alpha", "beta", "Villin", "Pin1", "peptide_binding"]


text_prompt_dict = {
    "alpha": [101, 201],
    "beta": [101, 201],
    # "Villin": [101, 102, 201, 202],
    # "Pin1": [101, 102, 201, 202],
    # "hYAP65": [101, 102, 201, 202],
    "Villin": [101, 201],
    "Pin1": [101, 201],
    "hYAP65": [101, 201],
    "peptide_binding": [101, 201],
}


def prase_hit_ratio(filename):
    hit, total, hit_ratio = None, None, None
    second_hit, second_total, second_hit_ratio = None, None, None
    try:
        f = open(filename, "r")
        while True:
            line = f.readline()
            if not line:
                break
            
            line = line.strip()
            if line.startswith("hit:"):
                hit = int(line.split(":")[1].strip())
            elif line.startswith("total:"):
                total = int(line.split(":")[1].strip())
            elif line.startswith("hit ratio:"):
                hit_ratio = float(line.split(":")[1].strip())

            elif line.startswith("eval hit:"):
                hit = int(line.split(":")[1].strip())
            elif line.startswith("eval total:"):
                total = int(line.split(":")[1].strip())
            elif line.startswith("eval hit ratio:"):
                hit_ratio = float(line.split(":")[1].strip())

            elif line.startswith("DSSP hit:"):
                second_hit = int(line.split(":")[1].strip())
            elif line.startswith("DSSP total:"):
                second_total = int(line.split(":")[1].strip())
            elif line.startswith("DSSP hit ratio:"):
                second_hit_ratio = float(line.split(":")[1].strip())

            elif line.startswith("pLDDT hit:"):
                second_hit = int(line.split(":")[1].strip())
            elif line.startswith("pLDDT total:"):
                second_total = int(line.split(":")[1].strip())
                assert total == second_total
            elif line.startswith("pLDDT hit ratio:"):
                second_hit_ratio = float(line.split(":")[1].strip())
    except:
        pass
    return hit, total, hit_ratio, second_hit, second_total, second_hit_ratio
