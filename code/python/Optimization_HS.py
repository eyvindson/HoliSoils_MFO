RCP = "RCP0" # no climate change
#filename = "DATA/DATA.h5" # Test data from Central Finland with 3579 forest stands
#scenario = "NFS"
extension = "test" # some additional info to the saved output 

#import wget
import os
import pandas as pd
import sys
from python import MultiFunc_HS as MFO
import numpy as np
    
from importlib import reload
reload(MFO)
module_path=os.path.abspath(os.path.join(os.path.dirname(os.path.join("")), '..'))+"//"

# patch to remove prints in voila build
IS_VOILA = os.environ.get("VOILA_SERVER_URL") is not None
def print_local(s):
    if not IS_VOILA:
        print(s)
    
class OptGUI:
    
    def __init__(self,scenario,defined,data):
        
        
        self.scenario = scenario
        self.data = data
        self.defined = defined
        self.filename = data
        
        path_to_zip_file = module_path + "data/compressed/" + self.data
        directory_to_extract_to = module_path+"data"

        try:
            import zipfile
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(directory_to_extract_to)
            print_local("Zip Extracted")
        except:
            print_local("Zip already existed")
        
        self.mfo = MFO.MultiFunctionalOptimization() 
        self.mfo.readData(module_path+"data/compressed/"+self.filename,areaCol="AREA",
                     sampleRatio=1.,delimeter=";",spatial="None"#"YLA.geojson" #If no sample ratio given, the ratio is assumed to be 1.
                    )
        self.columnTypes = {
            'TOTAL_SOIL_c  ':(float,"Relative to Area"), 
            'NBP_pool_c':(float,"Relative to Area"),
            'NPP':(float,"Relative to Area")
        }
        self.mfo.LOC = "HS"
        self.mfo.unit={"TOTAL_SOIL_c":"kg C ha<sup>-1</sup>",
                   "NBP_pool_c":"kg C ha <sup>-1</sup>",
                   "NPP":"??"}
        
        self.mfo.unit_head={"TOTAL_SOIL_c":"Total soil C",
                   "NBP_pool_c":"NBP pool",
                       "NPP":"Net primary production"}
        
        self.mfo.calculateTotalValuesFromRelativeValues(columnTypes=self.columnTypes)

        self.regimeClassNames = {"regimeClass0name":"CCF","regimeClass1name":"SA","regimeClass2name":"Broadleave"}
        self.regimeClassregimes = {"regimeClass0regimes":["CCF_3","CCF_4","BAUwGTR"],"regimeClass1regimes":["SA"],"regimeClass2regimes":["BAUwT_B", "BAUwT_5_B", "BAUwT_15_B", "BAUwT_30_B", "BAUwT_GTR_B"]}

        self.mfo.addRegimeClassifications(regimeClassNames = self.regimeClassNames,regimeClassregimes=self.regimeClassregimes)

        self.mfo.finalizeData(initialRegime="initial_state")

        
        if self.scenario == 'BDS':
            
            self.wood_production_bioenergy = { 
            
            "TOTAL_SOIL_c" : ["Total soil C",
                            "TOTAL_SOIL_c",
                            "max","lastYear","areaWeightedAverage","TEXT FOR HOVERING NPV"],
            
            # Harvested roundwood - maximise (even flow)
            #"Average_Harvested_V" : ["Hakkuukertymä (Tukki + Kuitu) (m3/ha/v)",
            #                         "Harvested_V",
            #                         "max","min","areaWeightedAverage","TEXT FOR HOVERING AVG Harv V"]
            #

            #"Discounted_NPV_3" : ["Diskointi revenue (€)",
            #                         "DISC_3",
            #                         "max","lastYear","areaWeightedAverage","TEXT FOR HOVERING NPV"],
            # 
            
            }
            
            #self.game = {
            # HSI moose - maximise       
            #"Sum_Total_HSI_MOOSE": ["Total habitat index for MOOSE (max average over all years)",
            #                       "Total_HSI_MOOSE",
            #                       "max","average","sum","TEXT FOR HOVERING HSI MOOSE"],
            # HSI hazel grouse - maximise
            #"Sum_Total_HAZEL_GROUSE": ["Total habitat index for HAZEL_GROUSE (max average over yrs)",
            #                       "Total_HAZEL_GROUSE",
            #                       "max","average","sum","TEXT FOR HOVERING AVG Hazel Grouse"],
            # HSI carpercaillie - maximise
            #"Sum_Total_CAPERCAILLIE": ["Total habitat index for CAPERCAILLIE (max average over yrs)",
            #                       "Total_CAPERCAILLIE",
            #                       "max","average","sum","TEXT FOR HOVERING AVG Capercaillie"]
            #}
            
            #self.recreation = {
            # Recreation index - maximise
            #"Sum_Total_Recreation" : ["Virkistysarvo ",
            #                          "Total_Recreation",
            #                          "max","min","sum",'Total_rec Hovering'],
            #}
            
            self.biodiversity = {

            "NBP_pool_c" : ["NBP pool",
                        "NBP_pool_c",
                        "max","lastYear","areaWeightedAverage", "TEXT FOR HOVERING Carbon"],
   
            # Deadwood - target 2050, increase by XX%
            "NPP" : ["Net primary production",
                                               "NPP",
                                               "max","lastYear","areaWeightedAverage","TEXT FOR HOVERING AVG DW _2050"],
            # Average age - target 2100, increase by XX%
            #"Average_Age": ["Metsäikä (kertaa, suhteessa vuoteen 2024)",
            #                                  "Relative_Age",
            #                                  "max","targetYear","sum",2115],
            # Deciduous tree volume - target 2050, increase by XX% 
            #"relative_prc_V_deciduous_2050": ["Lehtipuiden osuus vuoteen 2050 mennessä (kertaa suhteessa vuoteen 2024)",
            #                                  "Relative_Total_prc_V_deciduous",
            #                                  "max","targetYearWithSlope","sum",2050,"TEXT FOR HOVERING total % V deciduous"],

                }
            
            
            self.objectives = {
                      **self.wood_production_bioenergy,
                      **self.biodiversity
                      #**self.recreation
            }
            
            print_local("objectives for BDS loaded")
            
            
        self.initialValues = {"Total_i_Vm3":107*10**6 / 19,               # from National Forest Policy            
                         "Total_Harvested_V": 72.3*10**6 / 19,       # from National Forest Policy 
                         "Total_Biomass": 2.9*10**6 / 19,            # from National Forest Policy  
                         "Total_CARBON_SINK" : 34.1*10**6 / 19,      # from National Forest Policy  
                                                    
                         "SA_forests" : 0.106,     # from ForestStatistics 2018
                         "CCF_forests" : 0.015,    # from ForestStatistics 2018
                         "BAUwGTR_forests":0.015}  # from ForestStatistics 2018    
        if defined == True:
            self.OBJECTIVES()

    def OBJECTIVES(self):
    
        self.mfo.defineObjectives(self.objectives,initialValues = self.initialValues)

        #self.CCFregimes = [regime for regime in self.mfo.regimes if "CCF" in regime] + ["SA"]

        #self.constraintTypes = {"CCFonPeat":["Allowed regimes","Turvemetsissä käytetään vain jatkuvakasvatusta",self.CCFregimes,"PEAT"]}

        #self.mfo.defineConstraints(self.constraintTypes)

        self.mfo.calculateObjectiveRanges(debug=False)

        self.mfo.showGUI(debug=False)

        
##FIXING ISSUE WITH JUPYTER -- may not be needed once plotly.py gets updated.
### FIX ISSUE WITH WIDGET -- 
def fix_widget_error():
    """
    Fix FigureWidget - 'mapbox._derived' Value Error.
    Adopted from: https://github.com/plotly/plotly.py/issues/2570#issuecomment-738735816
    """
    import shutil
    import pkg_resources

    pkg_dir = os.path.dirname(pkg_resources.resource_filename("plotly", "plotly.py"))

    basedatatypesPath = os.path.join(pkg_dir, "basedatatypes.py")

    backup_file = basedatatypesPath.replace(".py", "_bk.py")
    shutil.copyfile(basedatatypesPath, backup_file)

    # read basedatatypes.py
    with open(basedatatypesPath, "r") as f:
        lines = f.read()

    find = "if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):"

    replace = """if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):
                if key_path_str == "mapbox._derived":
                    return"""

    # add new text
    lines = lines.replace(find, replace)

    # overwrite old 'basedatatypes.py'
    with open(basedatatypesPath, "w") as f:
        f.write(lines)

# fix_widget_error()
