import os
def fix_label(folder,file):
    with open(os.path.join(folder,file), "r") as f:
        lines = f.readlines()
        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                img, label = parts
                if label == "1":
                    label = "0"
                elif label == "2":
                    label = "1"
                elif label == "3":
                    label = "2"
                elif label == "4":
                    label = "3"
                elif label == "5":
                    label = "4"
                elif label == "6":
                    label = "5"
                
                fixed_lines.append(f"{img} {label}\n")
            else:
                # if line is malformed, keep it as it is
                fixed_lines.append(line)
    if file=='train.txt':
        file="train_fix.txt"
        with open(os.path.join(folder,file), "w") as f:
            f.seek(0)
            f.truncate()
            f.writelines(fixed_lines)
    elif file=='val.txt':
        file="val.txt"
        with open(os.path.join(folder,file), "w") as f:
            f.seek(0)
            f.truncate()
            f.writelines(fixed_lines)
    elif file=='test.txt':
        file="test_fix.txt"
        with open(os.path.join(folder,file), "w") as f:
            f.seek(0)
            f.truncate()
            f.writelines(fixed_lines)

if __name__=="__main__":
    fix_label("dataset/labels/train.txt")
    fix_label("dataset/labels/val.txt")
    fix_label("dataset/labels/test.txt")