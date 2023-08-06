import requests
import re
import json
import argparse
import os
from gooey import Gooey, GooeyParser

image = os.path.join(os.getcwd(),"image_df")
@Gooey(menu=[{'name': 'About', 'items': [{
                'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': 'Database Fetcher',
                'description': 'Fetch model\'s latest data from Civitai.',
                'version': '1.0',
                'copyright': '2023',
                'website': 'https://github.com/Faildes/CivitAI-ModelFetch-AutoScripterPlanner',
                'developer': 'Chattiori',
                'license': 'Gooey, Civitai'
            }]}],dump_build_config=True, program_name="Database Fetcher",image_dir=image, body_bg_color="#e7f5ff")
def main():
    data = {}
    parsey = GooeyParser(description="Fetch Model Datas From CivitAI")
    parser = parsey.add_argument_group('Settings')
    parser.add_argument("--query", type=str, help="Text Query, blank for None", default=None, required=False)
    parser.add_argument("--tag", type=str, help="Tag Search, blank for None", default=None, required=False)
    parser.add_argument("--username", type=str, help="Model Creator Username, blank for None", default=None, required=False)
    parser.add_argument("--type", type=str, help="Model type, blank for None", default=None, required=False)
    parser.add_argument("--sort", type=str, help="Sort results by, Defaults to None", default=None, required=False)
    parser.add_argument("--period", type=str, help="Period of Model, blank for None", default=None, required=False)
    parser.add_argument("--rating", type=float, help="Search by Model Ratings, Defaults to 0", default=0, required=False)
    parser.add_argument("--page", type=int, help="First Pagination offset, Defaults to 1", default=1, required=False)
    parser.add_argument("--limit", type=int, help="Limit items returned, Defaults to 100", default=100, required=False)
    parser.add_argument("--output", type=str, help="Output File Name without Extension, Defaults to dump", default="dump", required=False)
    parser.add_argument("--allow_xl", action="store_true", help="Allow SDXL Models", required=False)
    args = parsey.parse_args()
    output = f"{args.output}.json"

    def _request_models(
        limit=100,
        page=1,
        query=None,
        tag=None,
        username=None,
        sort=None,
        period=None,
        rating=None,
    ) -> dict:

        endpoint = "https://civitai.com/api/v1/models"
        params = {
            "limit": limit,
            "page": page,
            "query": query,
            "tag": tag,
            "username": username,
            "sort": sort,
            "period": period,
            "rating": rating,
        }
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return None


    def get_models(
        limit=100,
        page=1,
        query=None,
        tag=None,
        username=None,
        model_type=None,
        sort=None,
        period=None,
        rating=0,
        allow_xl=False,
        save:bool=False):

        data = _request_models(
            limit, page, query, tag, username, sort, period, None
        )
        # access the response metadata
        metadata = data["metadata"]  # totalItems,currentPage,pageSize,totalPages,nextPage

        # Create a list of Model objects from the data
        models = {}
        numod = 0
        rate = float(rating) if rating is not None else 0
        for it in range(len(data["items"])):
            timo = False
            item = data["items"][it]
            if item["type"] != model_type or not item["allowDerivatives"] or not item["allowDifferentLicense"]:
                continue
            gid = 0
            side = 91971975197519751975.018010
            muds = 500000.000000
            rej_model_pref = ["inpaint"]
            for kt in range(len(item["modelVersions"])):
                for qy in range(len(item["modelVersions"][kt]["files"])):
                    if item["modelVersions"][kt]["baseModel"] != "SD 1.5" and allow_xl is False:
                        continue
                    if ((int(item["stats"]["ratingCount"]) > 0) and (float(item["stats"]["rating"]) < rate)) or item["poi"]:
                        continue
                    if any([re.search(s, item["modelVersions"][kt]["name"], flags=re.IGNORECASE) for s in rej_model_pref]) or any([re.search(s, item["name"], flags=re.IGNORECASE) for s in rej_model_pref]):
                        continue
                    if int(item["modelVersions"][kt]["id"]) < gid:
                        continue
                    else:
                        gid = int(item["modelVersions"][kt]["id"])
                    if item["modelVersions"][kt]["files"][qy]["sizeKB"] >= side or item["modelVersions"][kt]["files"][qy]["sizeKB"] <= muds or item["modelVersions"][kt]["files"][qy]["type"] != "Model":
                        continue
                    else:
                        side = item["modelVersions"][kt]["files"][qy]["sizeKB"]
                    model = {
                        "name":item["name"],
                        "id":item["id"],
                        "tags":item["tags"],
                        "model_versions_id":item["modelVersions"][kt]["id"],
                        "allowNoCredit":item["allowNoCredit"],
                        "allowCommercialUse":item["allowCommercialUse"],
                        "creator_username":item["creator"]["username"],
                        "model_versions_id":item["modelVersions"][kt]["id"],
                        "model_versions_sha256":item["modelVersions"][kt]["files"][qy]["hashes"]["SHA256"].lower(),
                        "model_versions_name":item["modelVersions"][kt]["name"],
                        "model_versions_download_url":item["modelVersions"][kt]["downloadUrl"],
                        "model_type":item["modelVersions"][kt]["baseModel"],
                        "model_versions_files_size_kb":item["modelVersions"][kt]["files"][qy]["sizeKB"],
                        "model_versions_files_format":item["modelVersions"][kt]["files"][qy]["metadata"],
                    }
                    models[item["id"]] = model
                    timo = True
            if timo:
                numod += 1
                timo = False
        return metadata, models, numod

    if os.path.exists(os.path.join(os.getcwd(),output)):
        os.remove(os.path.join(os.getcwd(),output))
    passed_args = {}
    if args.query is not None:
        passed_args["query"]=args.query
    if args.tag is not None:
        passed_args["tag"]=args.tag
    if args.username is not None:
        passed_args["username"]=args.username
    if args.type is not None:
        passed_args["model_type"]=args.type
    if args.sort is not None:
        passed_args["sort"]=args.sort
    if args.period is not None:
        passed_args["period"]=args.period
    if args.rating is not None:
        passed_args["rating"]=args.rating
    if args.allow_xl:
        passed_args["allow_xl"]=True
    trem = f"Fetching Data to {output} using parameters:\n"
    trem += f"limit: {args.limit},\npage: {args.page},\nperiod: {args.period},\nquery:{args.query},\n"
    trem += f"rating: More than {args.rating} Stars,\nsort: {args.sort},\ntag: {args.tag},\n"
    trem += f"type: {args.type},\nusername: {args.username}\n\n"
    print(trem)
    pages = args.page + (int(args.limit) // 100)
    last_lim = int(args.limit) % 100
    i = args.page
    mert = True
    while mert:
        passed_args["page"]=str(i)
        passed_args["limit"]="100"
        met, mod, g = get_models(**passed_args)
        max_page = int(met["totalPages"])
        last_lim += (100 - g)
        pages += last_lim // 100
        if pages >= max_page:
            pages = max_page
        last_lim = last_lim % 100
        data.update(mod)
        print(f"Fetching {len(data)}/{args.limit}...")
        i+=1
        if i>pages:
            mert = False
    if last_lim != 0:
        passed_args["limit"]=str(last_lim)
        passed_args["page"]=str(i)
        met, mod, g = get_models(**passed_args)
        data.update(mod)
        
        with open(output,mode="a+") as hss:
            json.dump(data, hss, indent=4)
            print(f"Saved Datas of {len(data)} Models to {output}")
if __name__ == '__main__':
    main()
