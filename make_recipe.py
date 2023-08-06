from scripter import make_code
from gooey import Gooey, GooeyParser
import random
import os

image = os.path.join(os.getcwd(),"image_sc")
def plan_parse(plan):
    try:
        res = int(plan)
    except ValueError:
        if plan.startswith("LIST:"):
            get = plan.replace("LIST:","").replace(" ","").split(",")
            res = list(set(get))
        elif plan.startswith("RANDNUM:"):
            get = plan.replace("RANDNUM:","").replace(" ","").split("-")
            res = random.randint(int(get[0]),int(get[1]))
        elif plan.startswith("PLAN:"):
            res = {}
            get = plan.replace("PLAN:","").replace(" ","").split(",%")
            print(plan)
            for x in get:
                de = x.split(":",1)
                res[de[0]] = de[1]
    print(res)
    return res
@Gooey(menu=[{'name': 'About', 'items': [{
                'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': 'Merge Scripter',
                'description': 'Make merge line using json file.',
                'version': '1.0',
                'copyright': '2023',
                'website': 'https://github.com/Faildes/CivitAI-ModelFetch-AutoScripterPlanner',
                'developer': 'Chattiori',
                'license': 'Gooey, Civitai'
            }]}],dump_build_config=True, program_name="Merge Scripter",image_dir=image, body_bg_color="#e7f5ff")
def main():
    parsey = GooeyParser(description="Make merge line using json file")
    parser = parsey.add_argument_group('Settings')
    parser.add_argument("--Input", type=str, help="JSON file name without extension, defaults to dump", default="dump", required=False)
    parser.add_argument("--VAE", type=str, help="VAE's Name, defaults to vae.ext", default="vae.ext", required=False)
    parser.add_argument("--VAE_URL", type=str, help="VAE's URL, defaults to url", default="url", required=False)
    parser.add_argument("--Recipe", type=str, help="Recipe file name without extension, defaults to recipe", default="recipe", required=False)
    parser.add_argument("--Markdown", type=str, help="Credit file name without extension, defaults to merge", default="merge", required=False)
    parser.add_argument("--HTML", type=str, help="HTML version of markdown file's file name without extension, defaults to merge", default="merge", required=False)
    parser.add_argument("--Output", type=str, help="Final Name of the merge, defaults to merged", default="merged", required=False)
    parser.add_argument("--Plan", type=str, help="Merge recipe,Model list or Number of models, defaults to 10", default="10", required=False, widget="Textarea")
    parser.add_argument("--Least_Most", type=str, help="The max num and min num of ratios, defaults to 0.0-1.0", default="0.0-1.0", required=False)
    parser.add_argument("--Token", type=str, help="Token for Huggingface, defaults to YourToken", default="YourToken", required=False)
    parser.add_argument("--NameRepo", type=str, help="Text Query, defaults to Name/Repo", default="Name/Repo", required=False)
    args = parsey.parse_args()
    vae = args.VAE
    vae_url = args.VAE_URL
    output = args.Recipe + ".txt"
    output1 = args.Markdown + ".txt"
    output2 = args.HTML + ".html"
    final_name = args.Output
    plan = args.Plan
    numi = plan_parse(plan)
    abc = [float(x) for x in args.Least_Most.split("-")]
    inter=args.Input + ".json"
    safer=False
    rej_tag=None
    rej_name=None
    Token = args.Token
    NameRepo = args.NameRepo
    make_code(vae,vae_url,output,output1,output2,final_name,numi,abc,inter,safer,rej_tag,rej_name,Token,NameRepo)
if __name__ == '__main__':
    main()
