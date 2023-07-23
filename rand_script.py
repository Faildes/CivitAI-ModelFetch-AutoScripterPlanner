import json
import random
import os
import re
import numpy as np
import markdown as mk
import copy
md = mk.Markdown()

def calc_ratio(seed, low, high):
    np.random.seed(seed)
    ratios = np.random.uniform(low, high, (1, 26))
    ratios = ratios[0].tolist()
    rev_ratios = [1 - rat for rat in ratios]
    return rev_ratios, ratios

filedict = {}
fulldict = {}
pruned={}
models = {}
merged = {}
credit = {}
blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]

def calculate_size(mergition,inp):
    summed = 0.0
    merged = 0.0
    prune = 2082642.673828125
    prunes = False
    prune_id = []
    prune_id_k = []
    maxed = 60000000.000000000
    new_mergition = {}
    merge_line = {}
    merge_line_s = {}
    id_list = []
    id_list_k = []
    merged_id = []
    i = 0
    with open(inp,mode="r") as ss:
        k = json.load(ss)
        for output, merge in mergition.items():
            merge_line_s = merge
            ids = merge.split(",")[1].split("+")
            hut = 0.0
            merged_id.append(output)
            for ide in ids:
                if "TEMP" not in ide:
                    id_list_k.append(ide)
                    st = k.pop(str(ide))
                    size = st["model_versions_files_size_kb"]
                    if size >= 3000000.000000000:
                        prunes = True
                        prune_id_k.append(ide)
                    hut += size
                else:
                    if ide in merged_id:
                        merged_id.remove(ide)
            renp = summed + hut + prune
            if renp >= maxed:
                new_mergition[str(i)] = {
                    "id_list": id_list,
                    "prune_id": prune_id,
                    "merge_line": merge_line}
                i += 1
                summed = prune * len(merged_id)
                prunes = False
                prune_id = []
                id_list = []
                merge_line = {}
            summed += hut
            merge_line[output] = merge_line_s
            id_list.extend(id_list_k)
            prune_id.extend(prune_id_k)
            merge_line_s = {}
            prune_id_k = []
            id_list_k = []
        new_mergition[str(i)] = {
                "id_list": id_list,
                "prune_id": prune_id,
                "merge_line": merge_line}
    return new_mergition

def select_models(ids, intr, safe=False, reject_tag=None, rej_name=None):
    global filedict
    global fulldict
    global pruned
    global credit
    pr = 0
    h = 0.0
    with open(intr,mode="r") as ss:
        k = json.load(ss)
        for idm in ids:
            st = k.pop(str(idm))
            model_link = f"https://civitai.com/models/{str(idm)}/"
            name = st["name"]
            if rej_name is not None:
                if any([re.search(s, name, flags=re.IGNORECASE) for s in rej_name]):
                    continue
            tag=st["tags"]
            if reject_tag is not None:
                if type(reject_tag) is str:
                    if reject_tag in tag:
                        continue
                elif type(reject_tag) is list:
                    if set(tag) & set(reject_tag):
                        continue
            version = st["model_versions_name"]
            url = st["model_versions_download_url"]
            size = st["model_versions_files_size_kb"]
            if float(size) >= 7000000.000000000:
                continue
            h += float(size)
            form = st["model_versions_files_format"]["format"]
            if form != "SafeTensor" and safe:
                continue
            author = st["creator_username"]
            author_url = f"https://civitai.com/user/{author}"
            terf = f"* **[{name}-{version}]({model_link})** by **[{author}]({author_url})**\n"
            ori = f"{name}-{version}"
            credit[ori] = terf
            if form == "SafeTensor":
                safetensors = 1
            else:
                safetensors = 0
            name = re.sub(r"[ \(\)\'\"]","",name)
            version = re.sub(r"[ \(\)\'\"]","",version)
            if size >= 3000000.000000000:
                pr += 1
                pruned[f"{name}-{version}"] = [url,f"{name}-{version}-pruned",safetensors]
                safetensors = 1
                version = f"{version}-pruned"
            if safetensors == 1:
                filedict[f"{name}-{version}"] = [url, f"{name}-{version}.safetensors",model_link]
                fulldict[f"{idm}"] = [url, f"{name}-{version}.safetensors",model_link,ori]
            else:
                filedict[f"{name}-{version}"] = [url, f"{name}-{version}.ckpt",model_link]
                fulldict[f"{idm}"] = [url, f"{name}-{version}.ckpt",model_link,ori]

def make_dict(index, dicted, todict, final):
    exec_fin = False
    if len(dicted) >= 1:
        file_a = random.choice(list(dicted))
        dicted.remove(file_a)
        files = 1
    if len(dicted) >= 1:
        file_b = random.choice(list(dicted))
        mode = "WS"
        st_b = dicted.remove(file_b)
        files = 2
        seed_a = random.randint(1, 4294967295)
    if len(dicted) >= 1:
        file_c = random.choice(list(dicted))
        mode = "ST"
        st_c = dicted.remove(file_c)
        files = 3
        seed_b = random.randint(1, 4294967295)
    if len(dicted) == 0:
        exec_fin = True
        filename = final
    else:
        filename = f"TEMP_{index}"
        dicted.append(filename)
    if files == 2:
        todict[filename] = f"{mode},{file_a}+{file_b},0.0:1.0:{seed_a}"
    else:
        todict[filename] = f"{mode},{file_a}+{file_b}+{file_c},0.0:1.0:{seed_a}|0.0:1.0:{seed_b}"
    return todict, dicted, exec_fin
    
def select_models_rd(num, intr, safe=False, reject_tag=None, rej_name=None):
    global filedict
    global fulldict
    global pruned
    global credit
    i = 0
    pr = 0
    h = 0.0
    with open(intr,mode="r") as ss:
        k = json.load(ss)
        while i<num:
            rid = random.choice(list(k.keys()))
            st = k.pop(rid)
            model_link = f"https://civitai.com/models/{rid}/"
            name = st["name"]
            if rej_name is not None:
                if any([re.search(s, name, flags=re.IGNORECASE) for s in rej_name]):
                    continue
            tag=st["tags"]
            if reject_tag is not None:
                if type(reject_tag) is str:
                    if reject_tag in tag:
                        continue
                elif type(reject_tag) is list:
                    if set(tag) & set(reject_tag):
                        continue
            version = st["model_versions_name"]
            url = st["model_versions_download_url"]
            size = st["model_versions_files_size_kb"]
            if float(size) >= 3000000.000000000 and pr >= 3:
                continue
            if float(size) >= 7000000.000000000:
                continue
            h += float(size)
            form = st["model_versions_files_format"]["format"]
            if form != "SafeTensor" and safe:
                continue
            author = st["creator_username"]
            author_url = f"https://civitai.com/user/{author}"
            terf = f"* **[{name}-{version}]({model_link})** by **[{author}]({author_url})**\n"
            ori = f"{name}-{version}"
            credit[ori] = terf
            if form == "SafeTensor":
                safetensors = 1
            else:
                safetensors = 0
            name = re.sub(r"[ \(\)\'\"]","",name)
            version = re.sub(r"[ \(\)\'\"]","",version)
            if size >= 3000000.000000000:
                pr += 1
                pruned[f"{name}-{version}"] = [url,f"{name}-{version}-pruned",safetensors]
                safetensors = 1
                version = f"{version}-pruned"
            if safetensors == 1:
                filedict[f"{name}-{version}"] = [url, f"{name}-{version}.safetensors",model_link,ori]
                fulldict[f"{idm}"] = [url, f"{name}-{version}.safetensors",model_link,ori]
            else:
                filedict[f"{name}-{version}"] = [url, f"{name}-{version}.ckpt",model_link,ori]
                fulldict[f"{idm}"] = [url, f"{name}-{version}.ckpt",model_link,ori]
            i += 1

def make_script(vae, model_s, dicted, named, out, alpha, beta):
    global models
    global merged
    global fulldict
    file_a = None
    file_b = None
    file_c = None
    file_a_name = None
    file_b_name = None
    file_c_name = None
    exec_fin = False
    form_a = "!python merge.py "
    form_b = f"--vae \"/content/vae/{vae}\" "
    form_c = "\\\n--save_half --prune --save_safetensors --delete_source --output "
    forge = ""
    if len(model_s) >= 1:
        st_a = dicted.pop(model_s[0])
        file_a = st_a[1]
        file_a_name = st_a[3]
        link_a = st_a[2]
        files = 1
    if len(model_s) >= 2:
        st_b = dicted.pop(model_s[1])
        file_b = st_b[1]
        file_b_name = st_b[3]
        link_b = st_b[2]
        files = 2
    if len(model_s) >= 3:
        st_c = dicted.pop(model_s[2])
        file_c = st_c[1]
        file_c_name = st_c[3]
        link_c = st_c[2]
        files = 3
    filename = named
    dicted[filename] = [None, f"{filename}.safetensors",None,f"{filename}"]
    fulldict[filename] = [None, f"{filename}.safetensors",None,f"{filename}"]

    form = form_a
    if files == 2:
        form += f"\"WS\" \"/content/models/\" \\\n\"{file_a}\" \"{file_b}\" \\\n--m0_name \"{file_a_name}\" --m1_name \"{file_b_name}\" \\\n"
        stract_a = f"[{file_a_name}]({link_a})" if link_a is not None else f"{file_a_name}"
        stract_b = f"[{file_b_name}]({link_b})" if link_b is not None else f"{file_b_name}"
        forge += f"Weighted Sum, {stract_a} + {stract_b},"
        model0 = [file_a_name]
        if file_a_name in merged.keys():
            model0 = merged[file_a_name]
        else:
            models[file_a_name] = [1]*26
        model1 = [file_b_name]
        if file_b_name in merged.keys():
            model1 = merged[file_b_name]
        else:
            models[file_b_name] = [1]*26
    else:
        form += f"\"ST\" \"/content/models/\" \\\n\"{file_a}\" \"{file_b}\" --model_2 \"{file_c}\" \\\n--m0_name \"{file_a_name}\" --m1_name \"{file_b_name}\" --m2_name \"{file_c_name}\" \\\n"
        stract_a = f"[{file_a_name}]({link_a})" if link_a is not None else f"{file_a_name}"
        stract_b = f"[{file_b_name}]({link_b})" if link_b is not None else f"{file_b_name}"
        stract_c = f"[{file_c_name}]({link_c})" if link_c is not None else f"{file_c_name}"
        forge += f"Sum Twice, {stract_a} + {stract_b} + {stract_c},"
        model0 = [file_a_name]
        if file_a_name in merged.keys():
            model0 = merged[file_a_name]
        else:
            models[file_a_name] = [1]*26
        model1 = [file_b_name]
        if file_b_name in merged.keys():
            model1 = merged[file_b_name]
        else:
            models[file_b_name] = [1]*26
        model2 = [file_c_name]
        if file_c_name in merged.keys():
            model2 = merged[file_c_name]
        else:
            models[file_c_name] = [1]*26
    form += form_b
    if files >= 2:
        seed = int(alpha[2])
        form += f"\\\n--rand_alpha \"{alpha[0]}, {alpha[1]}, {seed}\" "
        forge += f"rand_alpha({alpha[0]}, {alpha[1]}, {seed}) "
        a0, a1 = calc_ratio(seed, float(alpha[0]), float(alpha[1]))
    if files == 3:
        seed = int(beta[2])
        form += f"--rand_beta \"{beta[0]}, {beta[1]}, {seed}\" "
        forge += f"rand_beta({beta[0]}, {beta[1]}, {seed}) "
        b0, b1 = calc_ratio(seed, float(beta[0]), float(beta[1]))
    form += form_c
    form += f"\"{filename}\"\n!pip cache purge\n\n"
    forge += f">> {filename}\n\n"
    if files == 3:
        for m in model0:
            models[m] = [n * a0[i] * b0[i] for i, n in enumerate(models[m])]
        for m in model1:
            models[m] = [n * a1[i] * b0[i] for i, n in enumerate(models[m])]
        for m in model2:
            models[m] = [n * b1[i] for i, n in enumerate(models[m])]
        merged[filename] = model0 + model1 + model2
    elif files == 2:
        for m in model0:
            models[m] = [n * a0[i] for i, n in enumerate(models[m])]
        for m in model1:
            models[m] = [n * a1[i] for i, n in enumerate(models[m])]
        merged[filename] = model0 + model1
    return form, dicted

def make_code(vae, vae_url, output, output1, output2, final_name, numi=None, inter=None, safer=False, rej_tag=None, rej_name=None,Token="YourToken",NameRepo="Name/Repo"):
    global filedict
    global fulldict
    global models
    global pruned
    global credit
    i = 0
    finale = False
    mdict = {}
    if os.path.exists(os.path.join(os.getcwd(),output)):
        os.remove(os.path.join(os.getcwd(),output))
    if os.path.exists(os.path.join(os.getcwd(),output1)):
        os.remove(os.path.join(os.getcwd(),output1))
    if os.path.exists(os.path.join(os.getcwd(),output2)):
        os.remove(os.path.join(os.getcwd(),output2))
    if type(numi) == int:
        select_models_rd(num=numi, intr=inter, safe=safer, reject_tag=rej_tag, rej_name=rej_name)
        count = numi
    elif type(numi) == list:
        select_models(ids=numi, intr=inter, safe=safer, reject_tag=rej_tag, rej_name=rej_name)
        count = len(numi)
    terfull = list(fulldict.keys())
    while finale is False:
        mdict, terfull, finale = make_dict(i, terfull, mdict, final_name)
        i += 1
    model_d = calculate_size(mdict, inter)
    with open(output1, mode="a+") as tr:
        tr.write("## Credits\n\n")
    with open(output, mode="a+", errors="ignore") as ou:
        with open("pt1.txt", mode="r", errors="ignore") as kts:
            erm = kts.read()
            while erm:
                erm = erm.replace("YourToken",Token)
                ou.write(erm)
                erm = kts.read()
        ou.write(f"\n!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \"{vae_url}\" -d \"/content/vae/\" -o \"{vae}\"\n%cd /content/merge-models/\n\n")
        for itere in model_d.keys():
            id_list = model_d[itere]["id_list"]
            prune_id = model_d[itere]["prune_id"]
            merge_line = model_d[itere]["merge_line"]
            pr_l = []
            for idew in id_list:
                base = fulldict[str(idew)]
                if idew in prune_id:
                    name = re.sub(r"[ \(\)\'\"]","",base[3])
                    pr_l.append(name)
                    basic = pruned[name]
                    tex = f"custom_model(\"{basic[0]}\", \"{basic[1]}\""
                    if basic[2] == 1:
                        tex += ",1)\n"
                    else:
                        tex += ")\n"
                else:
                    name = base[1].rsplit(".",1)[0]
                    tex = f"custom_model(\"{base[0]}\", \"{name}\""
                    if ".safetensors" in base[1]:
                        tex += ",1)\n"
                    else:
                        tex += ")\n"
                ou.write(tex)
            ou.write("\n")
            for key in pr_l:
                base = pruned[key]
                if base[2] == 1:
                    nam = f"{key}.safetensors"
                else:
                    nam = f"{key}.ckpt"
                form = f"!python merge.py \"NoIn\" \"/content/models/\" \\\n\"{nam}\" None \\\n--m0_name \"{key}\" \\\n--vae \"/content/vae/{vae}\" \\\n--save_half --prune --save_safetensors --delete_source --output \"{base[1]}\"\n!pip cache purge\n\n"
                ou.write(form)
            fullydict = copy.deepcopy(fulldict)
            for key, base in merge_line.items():
                splbase = base.split(",")
                moded = splbase[0]
                modelr = splbase[1].split("+")
                out_name = key
                alpha_beta = splbase[2].split("|")
                alpha = alpha_beta[0].split(":")
                beta = alpha_beta[1].split(":")
                scr, fullydict = make_script(vae, modelr, fullydict, out_name, output1, alpha, beta)
                ou.write(scr)
        fix = f"!python merge.py \"RM\" \"/content/models/\" \\\n\"{final_name}.safetensors\" None --output \"{final_name}-recipe\"\n!pip cache purge\n\n"
        fix+= "!pip install --upgrade huggingface_hub\n\nModel_Directory = \"/content/models/\" #@param {type:\"string\"}\n"
        fix += f"HF_File_Name = \"{final_name}\""
        fix += "#@param {type:\"string\"}\n"
        fix += f"Output_File = \"{final_name}.safetensors\"\nUpload_File = Model_Directory + Output_File\n"
        fix += f"User_Repository =\"{NameRepo}\""
        fix += " #@param {type:\"string\"}\n"
        fix += f"Output_File_1 = \"{final_name}-recipe.json\"\n"
        fix += "Upload_File_1 = f\"/content/merge-models/{Output_File_1}\"\n\n"
        fix += "%cd \{Model_Directory}\n\nfrom huggingface_hub import upload_file\nupload_file(path_or_fileobj=Upload_File, path_in_repo=Output_File, repo_id=User_Repository, token=Token)\nupload_file(path_or_fileobj=Upload_File_1, path_in_repo=Output_File_1, repo_id=User_Repository, token=Token)"
        ou.write(fix)
    print(f"Write mergition of {final_name} Using {count} models Successfully!")
    with open(output1, mode="a+") as op:
        for value in dict(sorted(credit.items())).values():
            op.write(value)
        op.write("\n\n## Mergition\n\n")
        for key, merge in mdict.items():
            splbase = merge.split(",")
            ids = splbase[1].split("+")
            out_name = key
            alpha_beta = splbase[2].split("|")
            alpha = alpha_beta[0].split(":")
            beta = alpha_beta[1].split(":")
            files = len(ids)
            forge = ""
            if len(ids) >= 1:
                st_a = fulldict[ids[0]]
                file_a = st_a[1]
                file_a_name = st_a[3]
                link_a = st_a[2]
                files = 1
            if len(ids) >= 2:
                st_b = fulldict[ids[1]]
                file_b = st_b[1]
                file_b_name = st_b[3]
                link_b = st_b[2]
                files = 2
            if len(ids) >= 3:
                st_c = fulldict[ids[2]]
                file_c = st_c[1]
                file_c_name = st_c[3]
                link_c = st_c[2]
                files = 3
            if files == 2:
                stract_a = f"[{file_a_name}]({link_a})" if link_a is not None else f"{file_a_name}"
                stract_b = f"[{file_b_name}]({link_b})" if link_b is not None else f"{file_b_name}"
                forge += f"* Weighted Sum, **{stract_a}** + **{stract_b}**,"
            else:
                stract_a = f"[{file_a_name}]({link_a})" if link_a is not None else f"{file_a_name}"
                stract_b = f"[{file_b_name}]({link_b})" if link_b is not None else f"{file_b_name}"
                stract_c = f"[{file_c_name}]({link_c})" if link_c is not None else f"{file_c_name}"
                forge += f"* Sum Twice, **{stract_a}** + **{stract_b}** + **{stract_c}**,"
            if files >= 2:
                seed = int(alpha[2])
                forge += f"rand_alpha(*{alpha[0]}, {alpha[1]}, {seed}*) "
            if files == 3:
                seed = int(beta[2])
                forge += f"rand_beta(*{beta[0]}, {beta[1]}, {seed}*) "
            forge += f">> **{out_name}**\n\n"
            op.write(forge)
    with open(output1, mode="a+") as tr:
        for k, l in models.items():
            kits = f"{k}: \n"
            ger = kits
            for i, n in enumerate(l):
                lp = n*100
                kits += f"{blockid[i]} = {lp:.02f}%, \n"
                ger += f"{blockid[i]} = {lp:.02f}%, "
            kits += "\n"
            ger += "\n"
            print(ger)
            tr.write(kits)
    with open(output2, mode="a+") as tr:
        with open(output1, mode="r") as rd:
            k = "\n".join(rd.readlines())
            te = md.convert(k)
            tr.write(te)
hsg = random.randint(10,27)

make_code("vae.ext", "vae url",\
          "script.txt","mergition.txt","merge.html","RandomAttempt",hsg, \
          "dump.json",Token="YourToken", NameRepo="Name/Repo")
