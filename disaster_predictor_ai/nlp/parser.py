import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class ClimateQueryParser:
    def __init__(self):
        self.locations = self._load_locations()
        self.hazard_synonyms = self._load_hazard_synonyms()
        self.time_patterns = self._load_time_patterns()
    
    def _load_locations(self) -> Dict[str, Dict]:
        return {
            # KENYA
            "nairobi": {"country": "Kenya", "type": "city", "region": "Central", "lat": -1.2921, "lon": 36.8219},
            "mombasa": {"country": "Kenya", "type": "city", "region": "Coast", "lat": -4.0435, "lon": 39.6682},
            "kisumu": {"country": "Kenya", "type": "city", "region": "Nyanza", "lat": -0.0917, "lon": 34.7680},
            "nakuru": {"country": "Kenya", "type": "city", "region": "Rift Valley", "lat": -0.3031, "lon": 36.0800},
            "eldoret": {"country": "Kenya", "type": "city", "region": "Rift Valley", "lat": 0.5143, "lon": 35.2698},
            "thika": {"country": "Kenya", "type": "city", "region": "Central", "lat": -1.0332, "lon": 37.0692},
            "malindi": {"country": "Kenya", "type": "city", "region": "Coast", "lat": -3.2175, "lon": 40.1169},
            "kitale": {"country": "Kenya", "type": "city", "region": "Rift Valley", "lat": 1.0157, "lon": 35.0062},
            "garissa": {"country": "Kenya", "type": "city", "region": "Eastern", "lat": -0.4536, "lon": 39.6401},
            "kakamega": {"country": "Kenya", "type": "city", "region": "Western", "lat": 0.2827, "lon": 34.7519},
            "turkana": {"country": "Kenya", "type": "county", "region": "Rift Valley", "lat": 3.1190, "lon": 35.5966},
            "marsabit": {"country": "Kenya", "type": "county", "region": "Eastern", "lat": 2.3284, "lon": 37.9899},
            "mandera": {"country": "Kenya", "type": "county", "region": "Eastern", "lat": 3.9366, "lon": 41.8670},
            "wajir": {"country": "Kenya", "type": "county", "region": "Eastern", "lat": 1.7471, "lon": 40.0629},
            "isiolo": {"country": "Kenya", "type": "county", "region": "Eastern", "lat": 0.3496, "lon": 37.5820},
            "samburu": {"country": "Kenya", "type": "county", "region": "Rift Valley", "lat": 1.5299, "lon": 37.1309},
            "west pokot": {"country": "Kenya", "type": "county", "region": "Rift Valley", "lat": 1.4109, "lon": 35.1178},
            "east pokot": {"country": "Kenya", "type": "county", "region": "Rift Valley", "lat": 1.2109, "lon": 35.7178},
            "baringo": {"country": "Kenya", "type": "county", "region": "Rift Valley", "lat": 0.4684, "lon": 35.9737},
            
            # SOUTH AFRICA
            "johannesburg": {"country": "South Africa", "type": "city", "province": "Gauteng", "lat": -26.2041, "lon": 28.0473},
            "cape town": {"country": "South Africa", "type": "city", "province": "Western Cape", "lat": -33.9249, "lon": 18.4241},
            "durban": {"country": "South Africa", "type": "city", "province": "KwaZulu-Natal", "lat": -29.8587, "lon": 31.0218},
            "pretoria": {"country": "South Africa", "type": "city", "province": "Gauteng", "lat": -25.7479, "lon": 28.2293},
            "port elizabeth": {"country": "South Africa", "type": "city", "province": "Eastern Cape", "lat": -33.9608, "lon": 25.6022},
            "bloemfontein": {"country": "South Africa", "type": "city", "province": "Free State", "lat": -29.0852, "lon": 26.1596},
            "east london": {"country": "South Africa", "type": "city", "province": "Eastern Cape", "lat": -32.9892, "lon": 27.8546},
            "pietermaritzburg": {"country": "South Africa", "type": "city", "province": "KwaZulu-Natal", "lat": -29.6020, "lon": 30.3794},
            "polokwane": {"country": "South Africa", "type": "city", "province": "Limpopo", "lat": -23.9045, "lon": 29.4689},
            "kimberley": {"country": "South Africa", "type": "city", "province": "Northern Cape", "lat": -28.7282, "lon": 24.7499},
            "nelspruit": {"country": "South Africa", "type": "city", "province": "Mpumalanga", "lat": -25.4753, "lon": 30.9700},
            "kwazulu natal": {"country": "South Africa", "type": "province", "lat": -28.5305, "lon": 30.8958},
            "gauteng": {"country": "South Africa", "type": "province", "lat": -26.2708, "lon": 28.1123},
            "western cape": {"country": "South Africa", "type": "province", "lat": -33.2277, "lon": 21.8569},
            "eastern cape": {"country": "South Africa", "type": "province", "lat": -32.2968, "lon": 26.4194},
            "limpopo": {"country": "South Africa", "type": "province", "lat": -23.4013, "lon": 29.4179},
            "mpumalanga": {"country": "South Africa", "type": "province", "lat": -25.5653, "lon": 30.5279},
            "free state": {"country": "South Africa", "type": "province", "lat": -28.4541, "lon": 26.7968},
            "northern cape": {"country": "South Africa", "type": "province", "lat": -29.0467, "lon": 21.8569},
            "north west": {"country": "South Africa", "type": "province", "lat": -26.6638, "lon": 25.2837},
            
            # NIGERIA
            "lagos": {"country": "Nigeria", "type": "city", "state": "Lagos", "lat": 6.5244, "lon": 3.3792},
            "abuja": {"country": "Nigeria", "type": "city", "state": "FCT", "lat": 9.0765, "lon": 7.3986},
            "kano": {"country": "Nigeria", "type": "city", "state": "Kano", "lat": 12.0022, "lon": 8.5920},
            "ibadan": {"country": "Nigeria", "type": "city", "state": "Oyo", "lat": 7.3775, "lon": 3.9470},
            "kaduna": {"country": "Nigeria", "type": "city", "state": "Kaduna", "lat": 10.5222, "lon": 7.4383},
            "port harcourt": {"country": "Nigeria", "type": "city", "state": "Rivers", "lat": 4.8156, "lon": 7.0498},
            "benin city": {"country": "Nigeria", "type": "city", "state": "Edo", "lat": 6.3350, "lon": 5.6037},
            "maiduguri": {"country": "Nigeria", "type": "city", "state": "Borno", "lat": 11.8311, "lon": 13.1511},
            "zaria": {"country": "Nigeria", "type": "city", "state": "Kaduna", "lat": 11.0804, "lon": 7.7006},
            "aba": {"country": "Nigeria", "type": "city", "state": "Abia", "lat": 5.1066, "lon": 7.3668},
            "jos": {"country": "Nigeria", "type": "city", "state": "Plateau", "lat": 9.8965, "lon": 8.8583},
            "ilorin": {"country": "Nigeria", "type": "city", "state": "Kwara", "lat": 8.5000, "lon": 4.5500},
            "onitsha": {"country": "Nigeria", "type": "city", "state": "Anambra", "lat": 6.1667, "lon": 6.7833},
            "abeokuta": {"country": "Nigeria", "type": "city", "state": "Ogun", "lat": 7.1500, "lon": 3.3500},
            "borno": {"country": "Nigeria", "type": "state", "lat": 11.8311, "lon": 13.1511},
            "adamawa": {"country": "Nigeria", "type": "state", "lat": 9.3265, "lon": 12.3984},
            "yobe": {"country": "Nigeria", "type": "state", "lat": 11.7467, "lon": 11.9668},
            "bauchi": {"country": "Nigeria", "type": "state", "lat": 10.3158, "lon": 9.8442},
            "gombe": {"country": "Nigeria", "type": "state", "lat": 10.2897, "lon": 11.1711},
            "plateau": {"country": "Nigeria", "type": "state", "lat": 9.2182, "lon": 9.5179},
            "taraba": {"country": "Nigeria", "type": "state", "lat": 8.8921, "lon": 11.3588},
            "nassarawa": {"country": "Nigeria", "type": "state", "lat": 8.5378, "lon": 8.3206},
            
            # ETHIOPIA
            "addis ababa": {"country": "Ethiopia", "type": "capital", "region": "Addis Ababa", "lat": 9.1450, "lon": 40.4897},
            "dire dawa": {"country": "Ethiopia", "type": "city", "region": "Dire Dawa", "lat": 9.5930, "lon": 41.8661},
            "mekelle": {"country": "Ethiopia", "type": "city", "region": "Tigray", "lat": 13.4967, "lon": 39.4721},
            "gondar": {"country": "Ethiopia", "type": "city", "region": "Amhara", "lat": 12.6089, "lon": 37.4671},
            "awasa": {"country": "Ethiopia", "type": "city", "region": "SNNP", "lat": 7.0621, "lon": 38.4769},
            "bahir dar": {"country": "Ethiopia", "type": "city", "region": "Amhara", "lat": 11.5941, "lon": 37.3907},
            "adama": {"country": "Ethiopia", "type": "city", "region": "Oromia", "lat": 8.5400, "lon": 39.2675},
            "jimma": {"country": "Ethiopia", "type": "city", "region": "Oromia", "lat": 7.6667, "lon": 36.8333},
            "somali region": {"country": "Ethiopia", "type": "region", "lat": 6.0000, "lon": 43.0000},
            "afar region": {"country": "Ethiopia", "type": "region", "lat": 11.7500, "lon": 40.5000},
            "tigray": {"country": "Ethiopia", "type": "region", "lat": 14.0000, "lon": 38.0000},
            "amhara": {"country": "Ethiopia", "type": "region", "lat": 11.0000, "lon": 38.0000},
            "oromia": {"country": "Ethiopia", "type": "region", "lat": 8.0000, "lon": 39.0000},
            
            # GHANA
            "accra": {"country": "Ghana", "type": "capital", "region": "Greater Accra", "lat": 5.6037, "lon": -0.1870},
            "kumasi": {"country": "Ghana", "type": "city", "region": "Ashanti", "lat": 6.6885, "lon": -1.6244},
            "tamale": {"country": "Ghana", "type": "city", "region": "Northern", "lat": 9.4008, "lon": -0.8393},
            "cape coast": {"country": "Ghana", "type": "city", "region": "Central", "lat": 5.1053, "lon": -1.2466},
            "sekondi takoradi": {"country": "Ghana", "type": "city", "region": "Western", "lat": 4.9344, "lon": -1.7133},
            "ho": {"country": "Ghana", "type": "city", "region": "Volta", "lat": 6.6108, "lon": 0.4708},
            "koforidua": {"country": "Ghana", "type": "city", "region": "Eastern", "lat": 6.0940, "lon": -0.2580},
            "sunyani": {"country": "Ghana", "type": "city", "region": "Brong Ahafo", "lat": 7.3397, "lon": -2.3264},
            "wa": {"country": "Ghana", "type": "city", "region": "Upper West", "lat": 10.0606, "lon": -2.5000},
            "bolgatanga": {"country": "Ghana", "type": "city", "region": "Upper East", "lat": 10.7856, "lon": -0.8514},
            
            # TANZANIA
            "dar es salaam": {"country": "Tanzania", "type": "city", "region": "Dar es Salaam", "lat": -6.7924, "lon": 39.2083},
            "dodoma": {"country": "Tanzania", "type": "capital", "region": "Dodoma", "lat": -6.1630, "lon": 35.7516},
            "mwanza": {"country": "Tanzania", "type": "city", "region": "Mwanza", "lat": -2.5164, "lon": 32.9175},
            "arusha": {"country": "Tanzania", "type": "city", "region": "Arusha", "lat": -3.3869, "lon": 36.6830},
            "mbeya": {"country": "Tanzania", "type": "city", "region": "Mbeya", "lat": -8.9094, "lon": 33.4607},
            "morogoro": {"country": "Tanzania", "type": "city", "region": "Morogoro", "lat": -6.8235, "lon": 37.6612},
            "tanga": {"country": "Tanzania", "type": "city", "region": "Tanga", "lat": -5.0917, "lon": 39.1014},
            "tabora": {"country": "Tanzania", "type": "city", "region": "Tabora", "lat": -5.0165, "lon": 32.8330},
            
            # UGANDA
            "kampala": {"country": "Uganda", "type": "capital", "region": "Central", "lat": 0.3476, "lon": 32.5825},
            "gulu": {"country": "Uganda", "type": "city", "region": "Northern", "lat": 2.7856, "lon": 32.2998},
            "lira": {"country": "Uganda", "type": "city", "region": "Northern", "lat": 2.2399, "lon": 32.8998},
            "mbarara": {"country": "Uganda", "type": "city", "region": "Western", "lat": 0.6023, "lon": 30.6476},
            "jinja": {"country": "Uganda", "type": "city", "region": "Eastern", "lat": 0.4244, "lon": 33.2042},
            "entebbe": {"country": "Uganda", "type": "city", "region": "Central", "lat": 0.0522, "lon": 32.4634},
            "fort portal": {"country": "Uganda", "type": "city", "region": "Western", "lat": 0.6710, "lon": 30.2757},
            "kasese": {"country": "Uganda", "type": "city", "region": "Western", "lat": 0.1833, "lon": 30.0833},
            "arua": {"country": "Uganda", "type": "city", "region": "Northern", "lat": 3.0202, "lon": 30.9115},
            "masaka": {"country": "Uganda", "type": "city", "region": "Central", "lat": 0.3340, "lon": 31.7340},
            
            # RWANDA
            "kigali": {"country": "Rwanda", "type": "capital", "region": "Kigali", "lat": -1.9441, "lon": 30.0619},
            "butare": {"country": "Rwanda", "type": "city", "region": "Southern", "lat": -2.5967, "lon": 29.7394},
            "gitarama": {"country": "Rwanda", "type": "city", "region": "Southern", "lat": -2.0781, "lon": 29.7567},
            "ruhengeri": {"country": "Rwanda", "type": "city", "region": "Northern", "lat": -1.4994, "lon": 29.6336},
            "gisenyi": {"country": "Rwanda", "type": "city", "region": "Western", "lat": -1.7056, "lon": 29.2514},
            
            # BURUNDI
            "bujumbura": {"country": "Burundi", "type": "capital", "region": "Bujumbura Mairie", "lat": -3.3614, "lon": 29.3599},
            "gitega": {"country": "Burundi", "type": "city", "region": "Gitega", "lat": -3.4271, "lon": 29.9246},
            
            # DEMOCRATIC REPUBLIC OF CONGO
            "kinshasa": {"country": "DR Congo", "type": "capital", "region": "Kinshasa", "lat": -4.4419, "lon": 15.2663},
            "lubumbashi": {"country": "DR Congo", "type": "city", "region": "Katanga", "lat": -11.6610, "lon": 27.4794},
            "mbuji mayi": {"country": "DR Congo", "type": "city", "region": "Kasai Oriental", "lat": -6.1500, "lon": 23.6000},
            "kisangani": {"country": "DR Congo", "type": "city", "region": "Orientale", "lat": 0.5167, "lon": 25.2000},
            "kananga": {"country": "DR Congo", "type": "city", "region": "Kasai Occidental", "lat": -5.8964, "lon": 22.4164},
            "bukavu": {"country": "DR Congo", "type": "city", "region": "South Kivu", "lat": -2.5078, "lon": 28.8612},
            "goma": {"country": "DR Congo", "type": "city", "region": "North Kivu", "lat": -1.6792, "lon": 29.2228},
            
            # ZAMBIA
            "lusaka": {"country": "Zambia", "type": "capital", "region": "Lusaka", "lat": -15.3875, "lon": 28.3228},
            "ndola": {"country": "Zambia", "type": "city", "region": "Copperbelt", "lat": -12.9587, "lon": 28.6366},
            "kitwe": {"country": "Zambia", "type": "city", "region": "Copperbelt", "lat": -12.8024, "lon": 28.2132},
            "kabwe": {"country": "Zambia", "type": "city", "region": "Central", "lat": -14.4469, "lon": 28.4464},
            "chingola": {"country": "Zambia", "type": "city", "region": "Copperbelt", "lat": -12.5289, "lon": 27.8642},
            "mufulira": {"country": "Zambia", "type": "city", "region": "Copperbelt", "lat": -12.5500, "lon": 28.2500},
            "livingstone": {"country": "Zambia", "type": "city", "region": "Southern", "lat": -17.8419, "lon": 25.8561},
            
            # ZIMBABWE
            "harare": {"country": "Zimbabwe", "type": "capital", "region": "Harare", "lat": -17.8252, "lon": 31.0335},
            "bulawayo": {"country": "Zimbabwe", "type": "city", "region": "Bulawayo", "lat": -20.1594, "lon": 28.5810},
            "chitungwiza": {"country": "Zimbabwe", "type": "city", "region": "Harare", "lat": -18.0130, "lon": 31.0747},
            "mutare": {"country": "Zimbabwe", "type": "city", "region": "Manicaland", "lat": -18.9707, "lon": 32.6473},
            "gweru": {"country": "Zimbabwe", "type": "city", "region": "Midlands", "lat": -19.4500, "lon": 29.8167},
            "kwekwe": {"country": "Zimbabwe", "type": "city", "region": "Midlands", "lat": -18.9167, "lon": 29.8167},
            "kadoma": {"country": "Zimbabwe", "type": "city", "region": "Mashonaland West", "lat": -18.3333, "lon": 29.9167},
            "masvingo": {"country": "Zimbabwe", "type": "city", "region": "Masvingo", "lat": -20.0636, "lon": 30.8276},
            
            # BOTSWANA
            "gaborone": {"country": "Botswana", "type": "capital", "region": "South East", "lat": -24.6282, "lon": 25.9231},
            "francistown": {"country": "Botswana", "type": "city", "region": "North East", "lat": -21.1671, "lon": 27.5144},
            "molepolole": {"country": "Botswana", "type": "city", "region": "Kweneng", "lat": -24.4167, "lon": 25.4833},
            "serowe": {"country": "Botswana", "type": "city", "region": "Central", "lat": -22.3833, "lon": 26.7167},
            "selibe phikwe": {"country": "Botswana", "type": "city", "region": "Central", "lat": -22.0000, "lon": 27.8667},
            "maun": {"country": "Botswana", "type": "city", "region": "North West", "lat": -19.9833, "lon": 23.4167},
            
            # NAMIBIA
            "windhoek": {"country": "Namibia", "type": "capital", "region": "Khomas", "lat": -22.5597, "lon": 17.0832},
            "swakopmund": {"country": "Namibia", "type": "city", "region": "Erongo", "lat": -22.6792, "lon": 14.5272},
            "walvis bay": {"country": "Namibia", "type": "city", "region": "Erongo", "lat": -22.9575, "lon": 14.5053},
            "oshakati": {"country": "Namibia", "type": "city", "region": "Oshana", "lat": -17.7889, "lon": 15.6964},
            "rundu": {"country": "Namibia", "type": "city", "region": "Kavango East", "lat": -17.9333, "lon": 19.7667},
            "ondangwa": {"country": "Namibia", "type": "city", "region": "Oshana", "lat": -17.9167, "lon": 15.9667},
            "rehoboth": {"country": "Namibia", "type": "city", "region": "Hardap", "lat": -23.3167, "lon": 17.0833},
            "katima mulilo": {"country": "Namibia", "type": "city", "region": "Zambezi", "lat": -17.5000, "lon": 24.2667},
            
            # MOZAMBIQUE
            "maputo": {"country": "Mozambique", "type": "capital", "region": "Maputo", "lat": -25.9692, "lon": 32.5732},
            "beira": {"country": "Mozambique", "type": "city", "region": "Sofala", "lat": -19.8436, "lon": 34.8389},
            "nampula": {"country": "Mozambique", "type": "city", "region": "Nampula", "lat": -15.1165, "lon": 39.2666},
            "chimoio": {"country": "Mozambique", "type": "city", "region": "Manica", "lat": -19.1164, "lon": 33.4838},
            "quelimane": {"country": "Mozambique", "type": "city", "region": "Zambezia", "lat": -17.8786, "lon": 36.8883},
            "tete": {"country": "Mozambique", "type": "city", "region": "Tete", "lat": -16.1564, "lon": 33.5867},
            "xai xai": {"country": "Mozambique", "type": "city", "region": "Gaza", "lat": -25.0519, "lon": 33.6444},
            "lichinga": {"country": "Mozambique", "type": "city", "region": "Niassa", "lat": -13.3133, "lon": 35.2406},
            "pemba": {"country": "Mozambique", "type": "city", "region": "Cabo Delgado", "lat": -12.9740, "lon": 40.5178},
            "inhambane": {"country": "Mozambique", "type": "city", "region": "Inhambane", "lat": -23.8650, "lon": 35.3833},
            
            # MADAGASCAR
            "antananarivo": {"country": "Madagascar", "type": "capital", "region": "Analamanga", "lat": -18.8792, "lon": 47.5079},
            "toamasina": {"country": "Madagascar", "type": "city", "region": "Atsinanana", "lat": -18.1492, "lon": 49.4022},
            "antsirabe": {"country": "Madagascar", "type": "city", "region": "Vakinankaratra", "lat": -19.8667, "lon": 47.0333},
            "fianarantsoa": {"country": "Madagascar", "type": "city", "region": "Haute Matsiatra", "lat": -21.4500, "lon": 47.0833},
            "mahajanga": {"country": "Madagascar", "type": "city", "region": "Boeny", "lat": -15.7167, "lon": 46.3167},
            "toliara": {"country": "Madagascar", "type": "city", "region": "Atsimo-Andrefana", "lat": -23.3500, "lon": 43.6667},
            "antsiranana": {"country": "Madagascar", "type": "city", "region": "Diana", "lat": -12.2787, "lon": 49.2917},
            
            # MALAWI
            "lilongwe": {"country": "Malawi", "type": "capital", "region": "Central", "lat": -13.9626, "lon": 33.7741},
            "blantyre": {"country": "Malawi", "type": "city", "region": "Southern", "lat": -15.7861, "lon": 35.0058},
            "mzuzu": {"country": "Malawi", "type": "city", "region": "Northern", "lat": -11.4547, "lon": 34.0059},
            "zomba": {"country": "Malawi", "type": "city", "region": "Southern", "lat": -15.3889, "lon": 35.3194},
            "kasungu": {"country": "Malawi", "type": "city", "region": "Central", "lat": -13.0333, "lon": 33.4833},
            "mangochi": {"country": "Malawi", "type": "city", "region": "Southern", "lat": -14.4784, "lon": 35.2644},
            "salima": {"country": "Malawi", "type": "city", "region": "Central", "lat": -13.7833, "lon": 34.4500},
            "karonga": {"country": "Malawi", "type": "city", "region": "Northern", "lat": -9.9333, "lon": 33.9333},
            
            # LESOTHO
            "maseru": {"country": "Lesotho", "type": "capital", "region": "Maseru", "lat": -29.3151, "lon": 27.4869},
            "teyateyaneng": {"country": "Lesotho", "type": "city", "region": "Berea", "lat": -29.1500, "lon": 27.7333},
            "mafeteng": {"country": "Lesotho", "type": "city", "region": "Mafeteng", "lat": -29.8167, "lon": 27.2333},
            "hlotse": {"country": "Lesotho", "type": "city", "region": "Leribe", "lat": -28.8333, "lon": 28.0500},
            "mohales hoek": {"country": "Lesotho", "type": "city", "region": "Mohale's Hoek", "lat": -30.1500, "lon": 27.4667},
            "qacha's nek": {"country": "Lesotho", "type": "city", "region": "Qacha's Nek", "lat": -30.1167, "lon": 28.6833},
            "quthing": {"country": "Lesotho", "type": "city", "region": "Quthing", "lat": -30.4000, "lon": 27.7167},
            "butha buthe": {"country": "Lesotho", "type": "city", "region": "Butha-Buthe", "lat": -28.7667, "lon": 28.2500},
            "mokhotlong": {"country": "Lesotho", "type": "city", "region": "Mokhotlong", "lat": -29.2833, "lon": 29.0667},
            "thaba tseka": {"country": "Lesotho", "type": "city", "region": "Thaba-Tseka", "lat": -29.5167, "lon": 28.6167},
            
            # SWAZILAND/ESWATINI
            "mbabane": {"country": "Eswatini", "type": "capital", "region": "Hhohho", "lat": -26.3054, "lon": 31.1367},
            "manzini": {"country": "Eswatini", "type": "city", "region": "Manzini", "lat": -26.4833, "lon": 31.3667},
            "lobamba": {"country": "Eswatini", "type": "city", "region": "Hhohho", "lat": -26.4500, "lon": 31.2000},
            "nhlangano": {"country": "Eswatini", "type": "city", "region": "Shiselweni", "lat": -26.8500, "lon": 31.2000},
            "piggs peak": {"country": "Eswatini", "type": "city", "region": "Hhohho", "lat": -25.9667, "lon": 31.2500},
            "siteki": {"country": "Eswatini", "type": "city", "region": "Lubombo", "lat": -26.4500, "lon": 31.9500},
            
            # REGIONAL DESIGNATIONS
            "east africa": {"type": "region", "countries": ["Kenya", "Uganda", "Tanzania", "Rwanda", "Burundi"]},
            "west africa": {"type": "region", "countries": ["Nigeria", "Ghana", "Senegal", "Mali", "Burkina Faso", "Ivory Coast"]},
            "southern africa": {"type": "region", "countries": ["South Africa", "Zimbabwe", "Botswana", "Namibia", "Zambia", "Malawi"]},
            "central africa": {"type": "region", "countries": ["DR Congo", "Central African Republic", "Chad", "Cameroon"]},
            "horn of africa": {"type": "region", "countries": ["Ethiopia", "Somalia", "Eritrea", "Djibouti"]},
            "sahel": {"type": "region", "countries": ["Mali", "Niger", "Chad", "Sudan", "Mauritania", "Burkina Faso"]},
            "maghreb": {"type": "region", "countries": ["Morocco", "Algeria", "Tunisia", "Libya"]},
            "nile valley": {"type": "region", "countries": ["Egypt", "Sudan", "South Sudan", "Ethiopia", "Uganda"]},
            "great lakes": {"type": "region", "countries": ["Uganda", "Kenya", "Tanzania", "Rwanda", "Burundi", "DR Congo"]},
            "kalahari": {"type": "region", "countries": ["Botswana", "South Africa", "Namibia"]},
            "congo basin": {"type": "region", "countries": ["DR Congo", "Central African Republic", "Cameroon", "Equatorial Guinea", "Gabon"]},
            
            # COUNTRIES
            "kenya": {"type": "country", "continent": "Africa", "lat": -0.0236, "lon": 37.9062},
            "south africa": {"type": "country", "continent": "Africa", "lat": -30.5595, "lon": 22.9375},
            "nigeria": {"type": "country", "continent": "Africa", "lat": 9.0820, "lon": 8.6753},
            "ethiopia": {"type": "country", "continent": "Africa", "lat": 9.1450, "lon": 40.4897},
            "ghana": {"type": "country", "continent": "Africa", "lat": 7.9465, "lon": -1.0232},
            "tanzania": {"type": "country", "continent": "Africa", "lat": -6.3690, "lon": 34.8888},
            "uganda": {"type": "country", "continent": "Africa", "lat": 1.3733, "lon": 32.2903},
            "zimbabwe": {"type": "country", "continent": "Africa", "lat": -19.0154, "lon": 29.1549},
            "zambia": {"type": "country", "continent": "Africa", "lat": -13.1339, "lon": 27.8493},
            "malawi": {"type": "country", "continent": "Africa", "lat": -13.2543, "lon": 34.3015},
            "mozambique": {"type": "country", "continent": "Africa", "lat": -18.6657, "lon": 35.5296},
            "botswana": {"type": "country", "continent": "Africa", "lat": -22.3285, "lon": 24.6849},
            "namibia": {"type": "country", "continent": "Africa", "lat": -22.9576, "lon": 18.4904},
            "rwanda": {"type": "country", "continent": "Africa", "lat": -1.9403, "lon": 29.8739},
            "burundi": {"type": "country", "continent": "Africa", "lat": -3.3731, "lon": 29.9189},
            "lesotho": {"type": "country", "continent": "Africa", "lat": -29.6100, "lon": 28.2336},
            "eswatini": {"type": "country", "continent": "Africa", "lat": -26.5225, "lon": 31.4659},
            "madagascar": {"type": "country", "continent": "Africa", "lat": -18.7669, "lon": 46.8691},
            "dr congo": {"type": "country", "continent": "Africa", "lat": -4.0383, "lon": 21.7587},
            "congo": {"type": "country", "continent": "Africa", "lat": -0.2280, "lon": 15.8277}
        }
    
    def _load_hazard_synonyms(self) -> Dict[str, List[str]]:
        return {
            "drought": [
                "drought", "dry spell", "water shortage", "arid conditions", "lack of rain",
                "water crisis", "dryness", "water scarcity", "dry conditions", "arid",
                "water stress", "moisture deficit", "precipitation deficit"
            ],
            "flood": [
                "flood", "flooding", "inundation", "overflow", "deluge", "flash flood",
                "river overflow", "heavy rain damage", "waterlogging", "submersion",
                "torrential rain", "monsoon flooding", "urban flooding"
            ],
            "hunger": [
                "hunger", "food insecurity", "malnutrition", "famine", "food shortage",
                "starvation", "food crisis", "undernourishment", "food scarcity",
                "nutritional deficit", "food access", "food availability"
            ],
            "crop": [
                "crop yield", "harvest", "production", "agricultural output", "farming yield",
                "crop production", "harvest output", "agricultural productivity", "yield",
                "farming production", "crop performance", "agricultural yield", "cultivation"
            ]
        }
    
    def _load_time_patterns(self) -> Dict[str, str]:
        current_year = datetime.now().year
        return {
            "next year": str(current_year + 1),
            "this year": str(current_year),
            "2025": "2025",
            "2026": "2026",
            "2027": "2027",
            "january": f"{current_year}-01",
            "february": f"{current_year}-02",
            "march": f"{current_year}-03",
            "april": f"{current_year}-04",
            "may": f"{current_year}-05",
            "june": f"{current_year}-06",
            "july": f"{current_year}-07",
            "august": f"{current_year}-08",
            "september": f"{current_year}-09",
            "october": f"{current_year}-10",
            "november": f"{current_year}-11",
            "december": f"{current_year}-12",
            "dry season": "dry_season",
            "wet season": "wet_season",
            "rainy season": "wet_season",
            "monsoon": "monsoon",
            "harvest time": "harvest",
            "planting season": "planting"
        }
    
    def parse(self, query: str) -> Dict:
        query_lower = query.lower().strip()
        
        hazard_result = self._extract_hazard(query_lower)
        location_result = self._extract_location(query_lower)
        time_result = self._extract_time(query_lower)
        
        confidence = self._calculate_confidence(hazard_result, location_result, time_result)
        
        return {
            "hazard": hazard_result["hazard"] if hazard_result else None,
            "location": location_result["location"] if location_result else None,
            "location_details": location_result["details"] if location_result else None,
            "time": time_result["time"] if time_result else None,
            "confidence": confidence,
            "parsed_successfully": confidence >= 70
        }
    
    def _extract_hazard(self, query: str) -> Optional[Dict]:
        for hazard_type, synonyms in self.hazard_synonyms.items():
            for synonym in synonyms:
                pattern = r'\b' + re.escape(synonym) + r'\b'
                if re.search(pattern, query):
                    return {
                        "hazard": hazard_type,
                        "matched_term": synonym,
                        "confidence": 100
                    }
        return None
    
    def _extract_location(self, query: str) -> Optional[Dict]:
        multi_word_locations = {k: v for k, v in self.locations.items() if len(k.split()) > 1}
        single_word_locations = {k: v for k, v in self.locations.items() if len(k.split()) == 1}
        
        for location_name in sorted(multi_word_locations.keys(), key=len, reverse=True):
            if location_name in query:
                return {
                    "location": location_name,
                    "details": multi_word_locations[location_name],
                    "confidence": 100
                }
        
        for location_name, details in single_word_locations.items():
            pattern = r'\b' + re.escape(location_name) + r'\b'
            if re.search(pattern, query):
                return {
                    "location": location_name,
                    "details": details,
                    "confidence": 100
                }
        
        return None
    
    def _extract_time(self, query: str) -> Optional[Dict]:
        for time_phrase, normalized_time in self.time_patterns.items():
            pattern = r'\b' + re.escape(time_phrase) + r'\b'
            if re.search(pattern, query):
                return {
                    "time": normalized_time,
                    "matched_phrase": time_phrase,
                    "confidence": 100
                }
        
        year_pattern = r'\b(20\d{2})\b'
        year_match = re.search(year_pattern, query)
        if year_match:
            return {
                "time": year_match.group(1),
                "matched_phrase": year_match.group(1),
                "confidence": 100
            }
        
        return None
    
    def _calculate_confidence(self, hazard_result: Optional[Dict], 
                           location_result: Optional[Dict], 
                           time_result: Optional[Dict]) -> int:
        confidence = 0
        
        if hazard_result:
            confidence += 40
        if location_result:
            confidence += 40
        if time_result:
            confidence += 20
        
        return min(confidence, 100)
    
    def get_location_suggestions(self, partial_input: str) -> List[Dict]:
        partial_lower = partial_input.lower()
        suggestions = []
        
        for location_name, details in self.locations.items():
            if location_name.startswith(partial_lower):
                suggestions.append({
                    "name": location_name,
                    "details": details,
                    "display_name": self._format_location_display(location_name, details)
                })
        
        return sorted(suggestions, key=lambda x: len(x["name"]))[:10]
    
    def _format_location_display(self, location_name: str, details: Dict) -> str:
        if details["type"] == "country":
            return f"{location_name.title()}"
        elif "country" in details:
            return f"{location_name.title()}, {details['country']}"
        else:
            return location_name.title()
    
    def get_available_hazards(self) -> List[str]:
        return list(self.hazard_synonyms.keys())
    
    def get_available_countries(self) -> List[str]:
        countries = set()
        for details in self.locations.values():
            if details.get("type") == "country":
                countries.add(details.get("country", "Unknown"))
            elif "country" in details:
                countries.add(details["country"])
        return sorted(list(countries))
