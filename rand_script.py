import json
import random
import os
import re

filedict = {}
pruned={}

def select_models(out, ids, intr, safe=False, reject_tag=None, rej_name=None):
    global filedict
    global pruned
    pr = 0
    h = 0.0
    with open(intr,mode="r") as ss:
        with open(out,mode="a+",errors="ignore") as oss:
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
                if h >= 70000000.000000000:
                    print("Size is big")
                    break
                form = st["model_versions_files_format"]["format"]
                if form != "SafeTensor" and safe:
                    continue
                author = st["creator_username"]
                author_url = f"https://civitai.com/user/{author}"
                terf = f"[{name}-{version}]({model_link}) by [{author}]({author_url})\n"
                oss.write(terf)
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
                else:
                    filedict[f"{name}-{version}"] = [url, f"{name}-{version}.ckpt",model_link]
            oss.write("\n\n")
            
def select_models_rd(out, num, intr, safe=False, reject_tag=None, rej_name=None):
    global filedict
    global pruned
    i = 0
    pr = 0
    h = 0.0
    with open(intr,mode="r") as ss:
        with open(out,mode="a+",errors="ignore") as oss:
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
                if h >= 70000000.000000000:
                    print("Size is big")
                    break
                form = st["model_versions_files_format"]["format"]
                if form != "SafeTensor" and safe:
                    continue
                author = st["creator_username"]
                author_url = f"https://civitai.com/user/{author}"
                terf = f"[{name}-{version}]({model_link}) by [{author}]({author_url})\n"
                oss.write(terf)
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
                else:
                    filedict[f"{name}-{version}"] = [url, f"{name}-{version}.ckpt",model_link]
                i += 1
            oss.write("\n\n")

def custom_model(link,name,safetensors=0,prune=False):
    global filedict
    global pruned
    if prune:
        pruned[name] = [link,f"{name}-pruned",safetensors]
        safetensors = 1
        name = f"{name}-pruned"
    if safetensors==1:
        filedict[name] = [link, f"{name}.safetensors"]
    else:
        filedict[name] = [link, f"{name}.ckpt"]

def make_script(vae, index, dicted, final, out):
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
    if len(dicted) >= 1:
        file_a_name = random.choice(list(dicted.keys()))
        st_a = dicted.pop(file_a_name)
        file_a = st_a[1]
        link_a = st_a[2]
        files = 1
    if len(dicted) >= 1:
        file_b_name = random.choice(list(dicted.keys()))
        st_b = dicted.pop(file_b_name)
        file_b = st_b[1]
        link_b = st_b[2]
        files = 2
    if len(dicted) >= 1:
        file_c_name = random.choice(list(dicted.keys()))
        st_c = dicted.pop(file_c_name)
        file_c = st_c[1]
        link_c = st_c[2]
        files = 3
    if len(dicted) == 0:
        exec_fin = True
        filename = final
    else:
        filename = f"TEMP_{index}"
        dicted[filename] = [None, f"{filename}.safetensors",None]

    form = form_a
    if files == 2:
        form += f"\"WS\" \"/content/models/\" \\\n\"{file_a}\" \"{file_b}\" \\\n--m0_name \"{file_a_name}\" --m1_name \"{file_b_name}\" \\\n"
        stract_a = f"[{file_a_name}]({link_a})" if link_a is not None else f"{file_a_name}"
        stract_b = f"[{file_b_name}]({link_b})" if link_b is not None else f"{file_b_name}"
        forge += f"Weighted Sum, {stract_a} + {stract_b},"
    else:
        form += f"\"ST\" \"/content/models/\" \\\n\"{file_a}\" \"{file_b}\" --model_2 \"{file_c}\" \\\n--m0_name \"{file_a_name}\" --m1_name \"{file_b_name}\" --m2_name \"{file_c_name}\" \\\n"
        stract_a = f"[{file_a_name}]({link_a})" if link_a is not None else f"{file_a_name}"
        stract_b = f"[{file_b_name}]({link_b})" if link_b is not None else f"{file_b_name}"
        stract_c = f"[{file_c_name}]({link_c})" if link_c is not None else f"{file_c_name}"
        forge += f"Sum Twice, {stract_a} + {stract_b} + {stract_c},"
    form += form_b
    if files >= 2:
        seed = random.randint(1, 4294967295)
        form += f"\\\n--rand_alpha \"0, 1, {seed}\" "
        forge += f"rand_alpha(0.0, 1.0, {seed}) "
    if files == 3:
        seed = random.randint(1, 4294967295)
        form += f"--rand_beta \"0, 1, {seed}\" "
        forge += f"rand_beta(0.0, 1.0, {seed}) "
    form += form_c
    form += f"\"{filename}\"\n!pip cache purge\n\n"
    forge += f">> {filename}\n\n"

    with open(out, mode="a+", errors="ignore") as op:
        op.write(forge)
    return form, exec_fin, dicted

def make_code(vae, vae_url, output, output1, final_name, numi=None, inter=None, safer=False, rej_tag=None, rej_name=None,Token="YourToken",NameRepo="Name/Repo"):
    global filedict
    i = 0
    finale = False
    if os.path.exists(os.path.join(os.getcwd(),output)):
        os.remove(os.path.join(os.getcwd(),output))
    if os.path.exists(os.path.join(os.getcwd(),output1)):
        os.remove(os.path.join(os.getcwd(),output1))
    if type(numi) == int:
        select_models_rd(output1,num=numi, intr=inter, safe=safer, reject_tag=rej_tag, rej_name=rej_name)
        count = numi
    elif type(numi) == list:
        select_models(output1,ids=numi, intr=inter, safe=safer, reject_tag=rej_tag, rej_name=rej_name)
        count = len(numi)
    with open(output, mode="a+", errors="ignore") as ou:
        with open("pt1.txt", mode="r", errors="ignore") as kts:
            erm = kts.read()
            while erm:
                erm = erm.replace("YourToken",Token)
                ou.write(erm)
                erm = kts.read()
        for key, base in filedict.items():
            if key.replace("-pruned","") in list(pruned.keys()):
                d = key.replace("-pruned","")
                basic = pruned[d]
                tex = f"custom_model(\"{basic[0]}\", \"{d}\""
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
        ou.write(f"\n!wget -c -O \"/content/vae/{vae}\" \"{vae_url}\"\n%cd /content/merge-models/\n\n")
        for key, base in pruned.items():
            if base[2] == 1:
                nam = f"{key}.safetensors"
            else:
                nam = f"{key}.ckpt"
            form = f"!python merge.py \"NoIn\" \"/content/models/\" \\\n\"{nam}\" None \\\n--m0_name \"{key}\" \\\n--vae \"/content/vae/{vae}\" \\\n--save_half --prune --save_safetensors --delete_source --output \"{base[1]}\"\n!pip cache purge\n\n"
            ou.write(form)
        while finale is False:
            scr, finale, filedict = make_script(vae, i, filedict, final_name, output1)
            ou.write(scr)
            i += 1
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
hsg = random.randint(10,27)

make_code("vae.ext", "vae url",\
          "scripts.txt","mergitions.txt","RandomAttempt",hsg, \
          "dump.json",Token="YourToken", NameRepo="Name/Repo")
