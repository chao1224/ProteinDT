import os


def extract(file_path, task):
    f_ = open(file_path, "r")

    if task in ["ss3", "ss8"]:
        for line in f_.readlines():
            line = line.strip()
            if line.startswith("cb513"):
                line = line.replace("{", "").replace("}", "").replace(":", ",")
                line = line.split(",")
                value = float(line[4])
                value = "{:.5f}".format(value)

    elif task == "contact":
        for line in f_.readlines():
            line = line.strip()
            if line.startswith("metrics"):
                line = line.replace("{", "").replace("}", "").replace(":", ",")
                line = line.split(",")
                value = float(line[3])
                value = "{:.5f}".format(value)

    elif task == "remote_homology":
        for line in f_.readlines():
            line = line.strip()
            if line.startswith("metrics_fold"):
                line = line.replace("{", "").replace("}", "").replace(":", ",")
                line = line.split(",")
                value = float(line[4])
                value = "{:.5f}".format(value)

    elif task == "fluorescence":
        for line in f_.readlines():
            line = line.strip()
            if line.startswith("metrics"):
                line = line.replace("{", "").replace("}", "").replace(":", ",")
                line = line.split(",")
                value = float(line[3])
                value = "{:.5f}".format(value)

    elif task == "stability":
        for line in f_.readlines():
            line = line.strip()
            if line.startswith("metrics"):
                line = line.replace("{", "").replace("}", "").replace(":", ",")
                line = line.split(",")
                value = float(line[3])
                value = "{:.5f}".format(value)

    return value


task2hyper = {
    "ss3": [
        "3-3e-5-5-2-8-0.08",
    ],
    "ss8": [
        "3-3e-5-5-2-8-0.08",
        # "3-3e-5-5-2-16-0.08",
    ],
    "contact": [
        # "3-3e-5-10-1-1-0.08",
        "3-3e-5-10-1-2-0.08",
    ],
    "remote_homology": [
        # "3-3e-5-10-1-64-0.08",
        "3-3e-5-10-8-8-0.08",
    ],
    "fluorescence": [
        "3-3e-5-25-4-16-0.0-True",
    ],
    "stability": [
        # "3-3e-5-1-2-16-0.08",
        # "3-3e-5-3-2-16-0.08",
        "3-3e-5-5-2-16-0.08",
    ],
}

if __name__ == "__main__":
    task_list = [
        "ss3",  "ss8", "contact", "remote_homology", "fluorescence", "stability",
    ]
    pretrained_model_list=[
        "ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5",
        "ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-InfoNCE-0.1-batch-9-gpu-8-epoch-5",
        # "ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10",
    ]

    for pretrained_mode in pretrained_model_list:
        row = "ProteinCLAP"

        for task in task_list:
            value_list = []
            for hyper in task2hyper[task]:
                file_path = os.path.join("../output", pretrained_mode, "downstream_TAPE", task, hyper, "result.txt")
                try:
                    value = extract(file_path, task)
                    value_list.append(value)
                except:
                    print("% missing {}".format(file_path))

            if len(value_list) > 0:
                optimal_value = max(value_list)
                print("task", task, value_list)
                row = "{} & {}".format(row, optimal_value)
            else:
                row = "{} & {}".format(row, "--")

        print("%", pretrained_mode)
        row += "\\\\"
        print(row)
        print()
        print()
        print()
