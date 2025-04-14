import pandas as pd
import numpy as np
#import wget
import os
import requests
from pathlib import Path
from zipfile import ZipFile
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from IPython.display import clear_output
from ipywidgets import Button, HBox, VBox,interact
from datetime import datetime
import matplotlib.pyplot as plt

from ortools.linear_solver import pywraplp

import itertools

import ipywidgets as widgets
from IPython.display import display, HTML

from tqdm import tqdm
import sys

import math

module_path = os.path.abspath(os.path.join(''))
module_path=os.path.abspath(os.path.join(os.path.dirname(os.path.join("")), '..'))+"//"

# conditional to hide things in Voila builds but not regular notebook view
IS_VOILA = os.environ.get("VOILA_SERVER_URL") is not None
def display_local(*args):
    """Call IPython.display.display only if running a notebook locally
    (i.e. not in Voila builds)."""
    if not IS_VOILA:
        display(*args)

class MultiFunctionalOptimization:

    data = pd.DataFrame()
    columnTypes = {}
    initialData = pd.DataFrame()

    regimes = list()
    years = list()
    standIds = list()

    standAreas = pd.DataFrame()
    peat = pd.DataFrame()

    constraints = dict()

    solutionCounter = 0

    debug = False

    # Open source solver, if commercial is not available
    def __init__(self,solver="CLP"):
        if solver == "CLP":
            self.solver = pywraplp.Solver('MultiForest Optimization Problem',
                            pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
            display_local("Using CLP")
        elif solver == "CPLEX":
            self.solver = pywraplp.Solver('MultiForest Optimization Problem',
                            pywraplp.Solver.CPLEX_LINEAR_PROGRAMMING)
            display_local("Using CPLEX")
        elif solver == "GLOP":
            self.solver = pywraplp.Solver('MultiForest Optimization Problem',
                            pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
            display_local("Using GLOP")
        elif solver == "Gurobi": ## Added by JK
            self.solver =  pywraplp.Solver_CreateSolver ("GUROBI_LINEAR_PROGRAMMING") ## Added by JK
            display_local("Using GUROBI") ## Added by JK
        else:
            display_local("Undefined solver, using CLP")     
            self.solver = pywraplp.Solver('MultiForest Optimization Problem',
                            pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
        

    def readData(self,filename,sampleRatio=1,delimeter=";",
                standsEnu = "id",regimesEnu = ["regime"],timeEnu = "year",
                areaCol = "represented_area_by_NFIplot",samplingSubsets = None,spatial = "YLA.geojson"):
        self.sampleRatio = sampleRatio
        self.areaCol = areaCol
        self.timeEnu = timeEnu
        self.standsEnu = standsEnu
        self.spatial = spatial
        self.filename = filename
        if self.spatial == "None":
            self.data = pd.read_csv(filename,delimiter=delimeter)
        else:
            self.data = pd.read_csv(filename,delimiter=delimeter)
            geodf = gpd.read_file(module_path + "data/compressed/"+self.spatial)
                    

            #Merging databases together - linking spatial information with simulated
            geometry = pd.DataFrame(list(set(self.data['id'])),geodf[['geometry','area']])
            geometry = geometry.reset_index()
            d = {'id': list(set(self.data['id'])),'AREA':geodf['area']}
            geometry = pd.DataFrame(d)

            self.data = self.data.reset_index().merge(geometry,right_on="id",left_on="id")
            self.data = self.data.drop('index',axis=1)
        self.standsEnu = standsEnu
        if len(regimesEnu) == 1:
            self.regimesEnu = regimesEnu[0]
        else:
            self.regimesEnu = "combinedRegime"
            self.data["combinedRegime"] = [""]*len(self.data)
            for i,colname in enumerate(regimesEnu):
                self.data["combinedRegime"]+=[colname]*len(self.data)
                self.data["combinedRegime"]+=self.data[colname].astype(str).values
                if i < len(regimesEnu)-1:
                    self.data["combinedRegime"]+=["_"]*len(self.data)
        self.data.replace(np.nan,0,inplace=True)
        if sampleRatio < 1:
            if samplingSubsets is None:
                n = int(len(set(self.data[self.standsEnu].values))*sampleRatio)
                #Setting seed to todays date: 18052021
                np.random.seed(18052021)
                stand_sample = np.random.choice(list(set(self.data[self.standsEnu].values)),n,replace=False)
                #display("sample size "+str(n)+"/"+str(len(set(self.data[self.standsEnu].values)))+"("+str(int(n/len(set(self.data[self.standsEnu].values))*100))+"%)")
            else:
                stand_sample = np.array([])
                for val in self.data[samplingSubsets].unique():
                    n = int(len(set(self.data[self.data[samplingSubsets] == val][self.standsEnu].values))*sampleRatio)
                    #Setting seed to todays date: 18052021
                    np.random.seed(18052021)
                    stand_sample = np.append(stand_sample,np.random.choice(list(set(self.data[self.data[samplingSubsets] == val][self.standsEnu].values)),n,replace=False))
            self.data = self.data[self.data[self.standsEnu].isin(stand_sample)]
            

    def CalculateTotalValues(self,**kwargs):
        self.columnTypes = kwargs
        for colname in self.data.columns:
            try:
                self.data[colname] = self.data[colname].astype(self.columnTypes[colname][0]).values
                if self.columnTypes[colname][1] == "Relative to Area":
                    self.data["Total_"+colname]=self.data[colname].values*self.data[self.areaCol].values
                elif self.columnTypes[colname][1] == "Relative to Volume":
                    self.data["Total_"+colname]=self.data[colname].values*self.data["V"].values
            except KeyError:
                if not colname in [self.timeEnu,self.regimesEnu,self.standsEnu]:
                    self.data[colname] = pd.to_numeric(self.data[colname],errors="ignore").values

    
    def calculateTotalValuesFromRelativeValues(self,columnTypes = dict()):
        if len(columnTypes) > 0: #If data about column types is given use that
            self.CalculateTotalValues(**columnTypes)
        else: ##If no data is given show a gui for selecting data
            colTypeChooser = widgets.interactive(self.CalculateTotalValues,{"manual":True},**{colname:["Absolute Value","Relative to Area","Relative to Volume"] for colname in self.data.columns})
            display(colTypeChooser)

    def addRegimes(self,**kwargs):
        for i in range(10):
            try:
                if len(kwargs["regimeClass"+str(i)+"name"])>0:
                    self.data[kwargs["regimeClass"+str(i)+"name"]+"_forests"] = self.data[self.regimesEnu].isin(kwargs["regimeClass"+str(i)+"regimes"])
            except KeyError:
                pass
    def addRegimeClassifications(self,regimeClassNames = dict(),regimeClassregimes=dict()):
        if len(regimeClassNames) > 0:
            self.addRegimes(**regimeClassNames,**regimeClassregimes)
        else:
            regimeClassificationChooser = widgets.interactive(self.addRegimes,{"manual":True},
            **{"regimeClass"+str(i)+"name":widgets.Text() for i in range(10)},
            **{"regimeClass"+str(i)+"regimes":widgets.SelectMultiple(options=tuple(self.data["regimesEnu"].unique())) for i in range(10)})
            display(regimeClassificationChooser)

    def finalizeData(self,
                initialRegime = "",initialTime = -np.inf):
        initialRequirement = np.array([True]*len(self.data))
        #print(self.data)
        if len(initialRegime) > 0:
            initialRequirement = initialRequirement*np.array(self.data[self.regimesEnu] == initialRegime)
        if initialTime > -np.inf:
            initialRequirement = initialRequirement*np.array(self.data[self.timeEnu] == initialTime)
        if len(initialRegime)>0 or initialTime>-np.inf:
            #Define initial data if such available
            self.initialData = self.data[initialRequirement]
            self.data = self.data[~initialRequirement]
            self.initialData.set_index([self.standsEnu,self.timeEnu,self.regimesEnu],inplace=True)
            self.initialData.sort_index(inplace=True)
            #print(initialRequirement)
            #print(self.data)
            #print(self.initialData)
            self.initialYear = min(self.initialData.index.get_level_values(self.timeEnu))
        self.data.set_index([self.standsEnu,self.timeEnu,self.regimesEnu],inplace=True)
        self.data.sort_index(inplace=True)
        #Use initialdata to define relative values if possible
        if len(initialRegime)>0 or initialTime>-np.inf: 
            for colname in self.data.columns:
                try:
                    if self.initialData.dtypes[colname] == float  and self.initialData[colname].sum()>0:
                        self.data["Relative_"+colname] = self.data[colname]/self.initialData[colname].sum()
                except (NameError,KeyError) as _:
                    pass
        self.regimes = self.data.index.get_level_values(self.regimesEnu).unique()
        
        self.years = self.data.index.get_level_values(self.timeEnu).unique()
        self.standIds = self.data.index.get_level_values(self.standsEnu).unique()

        self.standAreas = self.data.loc[(slice(None),self.years[0],slice(None)),self.areaCol]
        self.standAreas = self.standAreas.reset_index()
        self.standAreas.drop([self.regimesEnu,self.timeEnu],axis=1,inplace=True)
        self.standAreas.drop_duplicates(inplace=True)
        self.standAreas.set_index(self.standsEnu,inplace=True)
        self.standAreas.sort_index(inplace=True)


    def addConstraints(self,constraintTypes):
        self.constraintTypes = constraintTypes
        self.speciesValues = dict()
        for constraintName in self.constraintTypes.keys():
            if self.constraintTypes[constraintName][0] == "Allowed regimes":
                self.constraints[constraintName] = dict()
                for regime in self.regimes:
                    if regime not in self.constraintTypes[constraintName][2]:
                        for standId in self.standIds:
                            if (standId,regime) in self.regimesDecision.keys() and self.data.loc[(standId,self.years[0],regime),self.constraintTypes[constraintName][3]] == 1:
                                self.constraints[constraintName][(standId,regime)] = self.solver.Add(self.regimesDecision[(standId,regime)]<=1,name = "No"+regime+"with"+self.constraintTypes[constraintName][3]+"onStand"+str(standId))
            if self.constraintTypes[constraintName][0] == "Species reduction":
                speciesCol = self.constraintTypes[constraintName][2]
                periodNo = self.constraintTypes[constraintName][3]
                reductionAmount = self.constraintTypes[constraintName][4]
                self.constraints[constraintName] = dict()
                for i,year in enumerate(self.years):
                    self.speciesValues[(speciesCol,year)] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),speciesCol+"amountInyear"+str(year))
                    self.solver.Add(self.speciesValues[(speciesCol,year)] ==
                        sum(self.decisionFrame["Decision"].values*
                                        self.data.loc[(slice(None),year,slice(None)),speciesCol].values)/self.sampleRatio,
                        name="constraintforSpecies"+speciesCol+"InYear"+str(year))
                    if self.LOC == "NOR":
                        if i>= periodNo:
                            self.constraints[constraintName][year] = self.solver.Add(self.speciesValues[(speciesCol,year)]-(1-reductionAmount)*self.speciesValues[(speciesCol,year-periodNo*5)]>=-1e10)
                    else:
                        self.constraints[constraintName][year] = self.solver.Add(self.speciesValues[(speciesCol,year)]-(1-reductionAmount)*self.speciesValues[(speciesCol,year-periodNo)]>=-1e10)
            if self.constraintTypes[constraintName][0] == "Species increase":
                speciesCol = self.constraintTypes[constraintName][2]
                periodNo = self.constraintTypes[constraintName][3]
                reductionAmount = self.constraintTypes[constraintName][4]
                self.constraints[constraintName] = dict()
                display(self.constraintTypes[constraintName][0] )
                
                for i,year in enumerate(self.years):
                    self.speciesValues[(speciesCol,year)] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),speciesCol+"amountInyear"+str(year))
                    
                    self.solver.Add(self.speciesValues[(speciesCol,year)] ==
                        sum(self.decisionFrame["Decision"].values*
                                        self.data.loc[(slice(None),year,slice(None)),speciesCol].values)/self.sampleRatio,
                        name="constraintforSpeciesInc"+speciesCol+"InYear"+str(year))
                    if self.LOC == "NOR":
                        if i>= periodNo:
                            self.constraints[constraintName][year] = self.solver.Add(self.speciesValues[(speciesCol,year)]-(1+reductionAmount)*self.speciesValues[(speciesCol,year-periodNo*5)]<=-1e10,name = speciesCol + "valuePer"+str(periodNo)+"inYear"+str(year))
                    else:
                        self.constraints[constraintName][year] = self.solver.Add(self.speciesValues[(speciesCol,year)]-(1+reductionAmount)*self.speciesValues[(speciesCol,year-periodNo)]<=-1e10,name = speciesCol + "valuePer"+str(periodNo)+"inYear"+str(year))
                        
            if self.constraintTypes[constraintName][0] == "less than":
                if self.LOC =="SWE":
                    colname1 = self.constraintTypes[constraintName][2] 
                    colname2 = self.constraintTypes[constraintName][4] ## Changed back to 4 from 3 by JK
                    standWiseAggregation1 = self.constraintTypes[constraintName][3] ## Changed back to 3 from 4 by JK 
                    standWiseAggregation2 = self.constraintTypes[constraintName][5]
                    self.constraints[constraintName] = dict()
                    self.comparedValues = dict()
                                               
                    for year in self.years:
                        self.comparedValues[(colname1,year)] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),colname1+"amountInyear"+str(year))
                        self.comparedValues[(colname2,year)] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),colname2+"amountInyear"+str(year))
                        if standWiseAggregation1 == "sum":
                            # import pdb; pdb.set_trace()
                            if not "Relative_" in colname1:
                                self.solver.Add(self.comparedValues[(colname1,year)] ==
                                        sum(self.decisionFrame["Decision"].values*
                                                        self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values)/self.sampleRatio,
                                        name="constraintfor"+colname1+"comparisonInYear"+str(year))
                            else:
                                    self.solver.Add(self.comparedValues[(colname1,year)] ==
                                        sum(self.decisionFrame["Decision"].values*
                                                        self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values),
                                        name="constraintfor"+colname1+"comparisonInYear"+str(year))
                        elif standWiseAggregation1 == "areaWeightedAverage":
                            self.solver.Add(self.comparedValues[(colname1,year)]==
                                    sum(self.decisionFrame["Decision"].values*
                                                    (self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values))/self.standAreas.values.sum(),
                                    name="constraintfor"+colname1+"comparisonInYear"+str(year))
                        elif standWiseAggregation1 == "areaWeightedSum":
                            if not "Relative_" in colname1:
                                self.solver.Add(self.comparedValues[(colname1,year)] ==
                                sum(self.decisionFrame["Decision"].values*
                                                (self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values))/self.sampleRatio,
                                name="constraintfor"+colname1+"comparisonInYear"+str(year))
                            else:
                                self.solver.Add(self.comparedValues[(colname1,year)] ==
                                sum(self.decisionFrame["Decision"].values*
                                                (self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values)),
                                name="constraintfor"+colname1+"comparisonInYear"+str(year))
                        if standWiseAggregation2 == "sum":
                            # import pdb; pdb.set_trace()
                            if not "Relative_" in colname2:
                                self.solver.Add(self.comparedValues[(colname2,year)] ==
                                        sum(self.decisionFrame["Decision"].values*
                                                        self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values)/self.sampleRatio,
                                        name="constraintfor"+colname2+"comparisonInYear"+str(year))
                            else:
                                    self.solver.Add(self.comparedValues[(colname2,year)] ==
                                        sum(self.decisionFrame["Decision"].values*
                                                        self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values),
                                        name="constraintfor"+colname2+"comparisonInYear"+str(year))
                        elif standWiseAggregation2 == "areaWeightedAverage":
                            self.solver.Add(self.comparedValues[(colname2,year)]==
                                    sum(self.decisionFrame["Decision"].values*
                                                    (self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values))/self.standAreas.values.sum(),
                                    name="constraintfor"+colname2+"comparisonInYear"+str(year))
                        elif standWiseAggregation2 == "areaWeightedSum":
                            if not "Relative_" in colname2:
                                # import pdb; pdb.set_trace()
                                self.solver.Add(self.comparedValues[(colname2,year)] ==
                                sum(self.decisionFrame["Decision"].values*
                                                (self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values))/self.sampleRatio,
                                name="constraintfor"+colname2+"comparisonInYear"+str(year))
                            else:
                                self.solver.Add(self.comparedValues[(colname2,year)] ==
                                sum(self.decisionFrame["Decision"].values*
                                                (self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values)),
                                name="constraintfor"+colname2+"comparisonInYear"+str(year))
                        self.constraints[constraintName][year] = self.solver.Add(self.comparedValues[(colname1,year)]-self.comparedValues[(colname2,year)]<=10e10
                                ,name = colname1 + "lessThan"+colname2+"inYear"+str(year))
                else:
                    colname1 = self.constraintTypes[constraintName][2]
                    colname2 = self.constraintTypes[constraintName][3]
                    standWiseAggregation1 = self.constraintTypes[constraintName][4]
                    standWiseAggregation2 = self.constraintTypes[constraintName][5]
                    self.constraints[constraintName] = dict()
                    self.comparedValues = dict()
                    for regime in self.regimes:
                        for year in self.years:
                            self.comparedValues[(colname1,year)] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),colname1+"amountInyear"+str(year))
                            self.comparedValues[(colname2,year)] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),colname2+"amountInyear"+str(year))
                            if standWiseAggregation1 == "sum":
                                # import pdb; pdb.set_trace()
                                if not "Relative_" in colname1:
                                    self.solver.Add(self.comparedValues[(colname1,year)] ==
                                            sum(self.decisionFrame["Decision"].values*
                                                            self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values)/self.sampleRatio,
                                            name="constraintfor"+colname1+"comparisonInYear"+str(year))
                                else:
                                        self.solver.Add(self.comparedValues[(colname1,year)] ==
                                            sum(self.decisionFrame["Decision"].values*
                                                            self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values),
                                            name="constraintfor"+colname1+"comparisonInYear"+str(year))
                            elif standWiseAggregation1 == "areaWeightedAverage":
                                self.solver.Add(self.comparedValues[(colname1,year)]==
                                        sum(self.decisionFrame["Decision"].values*
                                                        (self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values))/self.standAreas.values.sum(),
                                        name="constraintfor"+colname1+"comparisonInYear"+str(year))
                            elif standWiseAggregation1 == "areaWeightedSum":
                                if not "Relative_" in colname1:
                                    self.solver.Add(self.comparedValues[(colname1,year)] ==
                                    sum(self.decisionFrame["Decision"].values*
                                                    (self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values))/self.sampleRatio,
                                    name="constraintfor"+colname1+"comparisonInYear"+str(year))
                                else:
                                    self.solver.Add(self.comparedValues[(colname1,year)] ==
                                    sum(self.decisionFrame["Decision"].values*
                                                    (self.data.loc[(slice(None),year,slice(None)),colname1].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values)),
                                    name="constraintfor"+colname1+"comparisonInYear"+str(year))
                            if standWiseAggregation2 == "sum":
                                # import pdb; pdb.set_trace()
                                if not "Relative_" in colname2:
                                    self.solver.Add(self.comparedValues[(colname2,year)] ==
                                            sum(self.decisionFrame["Decision"].values*
                                                            self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values)/self.sampleRatio,
                                            name="constraintfor"+colname2+"comparisonInYear"+str(year))
                                else:
                                        self.solver.Add(self.comparedValues[(colname2,year)] ==
                                            sum(self.decisionFrame["Decision"].values*
                                                            self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values),
                                            name="constraintfor"+colname2+"comparisonInYear"+str(year))
                            elif standWiseAggregation2 == "areaWeightedAverage":
                                self.solver.Add(self.comparedValues[(colname2,year)]==
                                        sum(self.decisionFrame["Decision"].values*
                                                        (self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values))/self.standAreas.values.sum(),
                                        name="constraintfor"+colname2+"comparisonInYear"+str(year))
                            elif standWiseAggregation2 == "areaWeightedSum":
                                if not "Relative_" in colname2:
                                    self.solver.Add(self.comparedValues[(colname2,year)] ==
                                    sum(self.decisionFrame["Decision"].values*
                                                    (self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values))/self.sampleRatio,
                                    name="constraintfor"+colname2+"comparisonInYear"+str(year))
                                else:
                                    self.solver.Add(self.comparedValues[(colname2,year)] ==
                                    sum(self.decisionFrame["Decision"].values*
                                                    (self.data.loc[(slice(None),year,slice(None)),colname2].sort_index().values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].sort_index().values)),
                                    name="constraintfor"+colname2+"comparisonInYear"+str(year))
                            self.constraints[constraintName][year] = self.solver.Add(self.comparedValues[(colname1,year)]-self.comparedValues[(colname2,year)]<=10e10
                                ,name = colname1 + "lessThan"+colname2+"inYear"+str(year))


    def defineConstraints(self,constraintTypes):
        if len(constraintTypes) > 0:
            self.addConstraints(constraintTypes)
        else:
            pass
            ## TODO think about GUI

    def addObjectives(self,objectiveTypes,initialValues):
        self.objectiveTypes = objectiveTypes

        display_local("Defining objectives")
                
        self.decisionFrame = pd.DataFrame(pd.Series(self.regimesDecision))
        self.decisionFrame.columns = ["Decision"]
        self.decisionFrame.sort_index(inplace=True)

        self.objectivesByYear = dict()
        for objName in self.objectiveTypes.keys():
            for year in self.years:
                self.objectivesByYear[(objName,year)] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),objName+"year"+str(year))

        self.maxDummyConstraints = {objName:
                        self.solver.Constraint(-self.solver.infinity(),self.solver.infinity(),"maxDummyConstraintfor"+objName)
                        for objName in self.objectiveTypes.keys()}
        self.maxDummy = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),"DummyVar")
        self.initialValues = dict()

        display_local("Aggregating stand wise")
        # Stand wise aggregation
        #for objName in tqdm(self.objectiveTypes.keys()): #With progress bar ?
        for objName in self.objectiveTypes.keys():
            if self.objectiveTypes[objName][4] == "sum":
                for year in self.years:
                    if not "Relative_" in self.objectiveTypes[objName][1]:
                        self.solver.Add(self.objectivesByYear[(objName,year)] ==
                                sum(self.decisionFrame["Decision"].values*
                                                self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][1]].values)/self.sampleRatio,
                                name="constraintfor"+objName+"InYear"+str(year))
                    else:
                            self.solver.Add(self.objectivesByYear[(objName,year)] ==
                                sum(self.decisionFrame["Decision"].values*
                                                self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][1]].values),
                                name="constraintfor"+objName+"InYear"+str(year))
                try:
                    if "Relative_" in self.objectiveTypes[objName][1]:
                        self.initialValues[objName] = 1
                    else:
                        try:
                            self.initialValues[objName] = initialValues[objName]
                        except KeyError:
                            self.initialValues[objName] = self.initialData.loc[(slice(None),slice(None),slice(None)),self.objectiveTypes[objName][1]].values.sum()
                except NameError:
                    pass
            elif self.objectiveTypes[objName][4] == "areaWeightedAverage":
                for year in self.years:
                    #print(year)
                    #print(len(self.regimesDecision))
                    #print(self.data['regime'].unique())
                    #print(len(self.decisionFrame["Decision"].values))
                    #print(len(self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][1]].values))
                    #print(len(self.data.loc[(slice(None),year,slice(None)),self.areaCol].values))
                    self.solver.Add(self.objectivesByYear[(objName,year)]==
                            sum(self.decisionFrame["Decision"].values*
                                            (self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][1]].values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].values))/self.standAreas.values.sum(),
                            name="constraintfor"+objName+"InYear"+str(year))
                try:

                    if "Relative_" in self.objectiveTypes[objName][1]:
                        self.initialValues[objName] = 1
                    else:
                        try:
                            self.initialValues[objName] = initialValues[objName]
                        except KeyError:
                            self.initialValues[objName] = self.initialData.loc[(slice(None),slice(None),slice(None)),self.objectiveTypes[objName][1]].values.sum()
                except NameError:
                    pass
            elif self.objectiveTypes[objName][4] == "areaWeightedSum":
                for year in self.years:
                        if not "Relative_" in self.objectiveTypes[objName][1]:
                            self.solver.Add(self.objectivesByYear[(objName,year)] ==
                            sum(self.decisionFrame["Decision"].values*
                                            (self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][1]].values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].values))/self.sampleRatio,
                            name="constraintfor"+objName+"InYear"+str(year))
                        else:
                            self.solver.Add(self.objectivesByYear[(objName,year)] ==
                            sum(self.decisionFrame["Decision"].values*
                                            (self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][1]].values*self.data.loc[(slice(None),year,slice(None)),self.areaCol].values)),
                            name="constraintfor"+objName+"InYear"+str(year))
  
                try:
                    if "Relative_" in self.objectiveTypes[objName][1]:
                        self.initialValues[objName] = 1
                    else:
                        try:
                            self.initialValues[objName] = initialValues[objName]
                        except KeyError:
                            self.initialValues[objName] = self.initialData.loc[(slice(None),slice(None),slice(None)),self.objectiveTypes[objName][1]].values.sum()
                except NameError:
                    pass
            elif self.objectiveTypes[objName][4] == "subsetSum":
                for year in self.years:
                    if not "Relative_" in self.objectiveTypes[objName][1]:
                        self.solver.Add(self.objectivesByYear[(objName,year)] ==
                                sum(self.decisionFrame["Decision"].values*
                                                (self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][1]].values*self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][5]].values))/self.sampleRatio,
                                name="constraintfor"+objName+"InYear"+str(year))
                    else:
                        self.solver.Add(self.objectivesByYear[(objName,year)] ==
                                sum(self.decisionFrame["Decision"].values*
f                                                (self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][1]].values*self.data.loc[(slice(None),year,slice(None)),self.objectiveTypes[objName][5]].values)),
                                name="constraintfor"+objName+"InYear"+str(year))
                try:
                    if "Relative_" in self.objectiveTypes[objName][1]:
                        self.initialValues[objName] = 1
                    else:
                        try:
                            self.initialValues[objName] = initialValues[objName]
                        except KeyError:
                            self.initialValues[objName] = self.initialData.loc[(slice(None),slice(None),slice(None)),self.objectiveTypes[objName][1]].values.sum()
                except NameError:
                    pass           

        self.objective = {
            objShortName:self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),objShortName)
            for objShortName in self.objectiveTypes.keys()
        }

        display_local("Aggregating year wise")
        # Year wise aggregation
        #for objName in tqdm(self.objectiveTypes.keys(),file=sys.stdout):
        for objName in self.objectiveTypes.keys():
            if self.objectiveTypes[objName][3] == "min":
                for year in self.years:
                    self.solver.Add(self.objective[objName]<=self.objectivesByYear[(objName,year)])
            elif self.objectiveTypes[objName][3] == "max":
                for year in self.years:
                    self.solver.Add(self.objective[objName]>=self.objectivesByYear[(objName,year)])
            elif self.objectiveTypes[objName][3] == "average":
                self.solver.Add(self.objective[objName]==sum([self.objectivesByYear[(objName,year)] for year in self.years])/len(self.years))
            elif self.objectiveTypes[objName][3] == "firstYear":
                self.solver.Add(self.objective[objName]==self.objectivesByYear[(objName,self.years[0])])
            elif self.objectiveTypes[objName][3] == "lastYear":
                self.solver.Add(self.objective[objName]==self.objectivesByYear[(objName,self.years[-1])])
            elif self.objectiveTypes[objName][3] == "sum":
                self.solver.Add(self.objective[objName]==sum([self.objectivesByYear[(objName,year)] for year in self.years]))
            elif self.objectiveTypes[objName][3] == "minYearlyIncrease":
                for i,year in enumerate(self.years):
                    if i>=1:
                        self.solver.Add(self.objective[objName]<=self.objectivesByYear[(objName,self.years[i])]-self.objectivesByYear[(objName,self.years[i-1])])
            elif self.objectiveTypes[objName][3] == "maxYearlyIncrease":
                for i,year in enumerate(self.years):
                    if i>=1:
                        self.solver.Add(self.objective[objName]>=self.objectivesByYear[(objName,self.years[i])]-self.objectivesByYear[(objName,self.years[i-1])])
            elif self.objectiveTypes[objName][3] == "maxDecreaseDuringNPeriods":
                for i,year in enumerate(self.years):
                    if i>=self.objectiveTypes[objName][5]:
                        self.solver.Add(self.objective[objName]>=self.objectivesByYear[(objName,self.years[i])]-self.objectivesByYear[(objName,self.years[i-self.objectiveTypes[objName][5]])])
            elif self.objectiveTypes[objName][3] == "minIncreaseDuringNPeriods":
                for i,year in enumerate(self.years):
                    if i>=self.objectiveTypes[objName][5]:
                        self.solver.Add(self.objective[objName]<=self.objectivesByYear[(objName,self.years[i])]-self.objectivesByYear[(objName,self.years[i-self.objectiveTypes[objName][5]])])
            elif self.objectiveTypes[objName][3] == "targetYearWithSlope":
                targetYear = self.objectiveTypes[objName][5]
                for year in self.years:
                    if year <= targetYear:
                        coeff = float(year-self.initialYear)/float(targetYear-self.initialYear)
                        self.solver.Add(self.objectivesByYear[(objName,year)] >= (1-coeff)*self.initialValues[objName]+coeff*self.objective[objName])
                    if year > targetYear:
                        self.solver.Add(self.objectivesByYear[(objName,year)] >= self.objective[objName] )
            elif self.objectiveTypes[objName][3] == "targetYear":
                targetYear = self.objectiveTypes[objName][5]
                for i,year in enumerate(self.years):
                    if self.LOC == "NOR":
                        if (i == 0 and year >targetYear) or (self.years[i-1] < targetYear and year > targetYear):
                            self.solver.Add(self.objectivesByYear[(objName,year)] == self.objective[objName] )
                    else:
                        if year > targetYear:
                            self.solver.Add(self.objectivesByYear[(objName,year)] >= self.objective[objName] )
            elif self.objectiveTypes[objName][3] == "periodicTargets":
                periodicTargets = self.objectiveTypes[objName][5]
                for i,year in enumerate(self.years):
                    self.solver.Add(self.objective[objName] <= (self.objectivesByYear[(objName,year)]-periodicTargets[i])/periodicTargets[i])
            else:
                display("Undefined yearly aggregation "+self.objectiveTypes[objName][3])
        display_local("Objectives added")

    def defineObjectives(self,objectiveTypes,initialValues=dict()):

        self.regimesDecision = {(standId,regime):
                   self.solver.IntVar(0,1,"treatmentDecisionForStand"+str(standId)+"Regime"+str(regime)) 
                   #self.solver.NumVar(0,1,"treatmentDecisionForStand"+str(standId)+"Regime"+regime) 
                   for (standId,regime) in itertools.product(self.standIds,self.regimes) if (standId,self.years[0],regime) in self.data.index}
        for standId in self.standIds:
            self.solver.Add(sum([self.regimesDecision[(standId,regime)] for regime in self.regimes if (standId,self.years[0],regime) in self.data.index])==1,name = "regimeConstraintForStand"+str(standId))
        if len(objectiveTypes) > 0:
            self.addObjectives(objectiveTypes,initialValues)
        else:
            pass
            ## TODO think about GUI

    def addGlobiomTargets(self,targetDict,transferRates,exactMatching=False):
        #TODO check that addobjectives has been run so that self.objectiveTypes exists
        sources = list(transferRates.keys())
        self.globiomProduction = dict()
        self.globiomTargets = targetDict
        for source in sources:
            self.globiomProduction[source] = dict()
            for year in self.years:
                self.globiomProduction[source][year] = dict()
                for target in transferRates[source].keys():
                   self.globiomProduction[source][year][target] = self.solver.NumVar(0,self.solver.infinity(),name=source+"UsedForGlobiomTarget"+target+"inYear"+str(year))
        self.globiomConstraint = dict()
        for source in sources:
            self.globiomConstraint[source] = dict()
            for year in self.years:
                targets = transferRates[source].keys()
                self.globiomConstraint[source][year] = self.solver.Add(sum(self.globiomProduction[source][year][target] for target in targets) 
                == sum(self.decisionFrame["Decision"].values*self.data.loc[(slice(None),year,slice(None)),source].sort_index().values)/self.sampleRatio,
                name = "GlobiomConstraintforsource"+source+"InYear"+str(year))
        targets = targetDict.keys()
        for target in targets:
            self.objectiveTypes["GlobiomTargetFor"+target] = [""]*5
            self.objectiveTypes["GlobiomTargetFor"+target][0] = "Relative meeting of globiom target for "+target
            if not exactMatching:
                self.objectiveTypes["GlobiomTargetFor"+target][2] = "max"
            else:
                self.objectiveTypes["GlobiomTargetFor"+target][2] = "min"
            self.objective["GlobiomTargetFor"+target] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),"GlobiomTargetFor"+target)
            self.maxDummyConstraints["GlobiomTargetFor"+target] = self.solver.Constraint(-self.solver.infinity(),self.solver.infinity(),"maxDummyConstraintforGlobiomTargetFor"+target)
            for i,year in enumerate(self.years):
                if not exactMatching:
                    self.solver.Add(self.objective["GlobiomTargetFor"+target]<= 
                    sum(np.array([self.globiomProduction[source][year][target] for source in sources if target in transferRates[source].keys()])*
                    np.array([transferRates[source][target][0] for source in sources if target in transferRates[source].keys()]))
                    /targetDict[target][i],
                    name="GlobiomConstraintforTarget"+target)
                else:
                    self.solver.Add(self.objective["GlobiomTargetFor"+target]>= 
                    sum(np.array([self.globiomProduction[source][year][target] for source in sources if target in transferRates[source].keys()])*
                    np.array([transferRates[source][target][0] for source in sources if target in transferRates[source].keys()]))-targetDict[target][i],
                    name="GlobiomConstraint1forTarget"+target)
                    self.solver.Add(self.objective["GlobiomTargetFor"+target]>= 
                    targetDict[target][i]-sum(np.array([self.globiomProduction[source][year][target] for source in sources if target in transferRates[source].keys()])*np.array([transferRates[source][target][0] for source in sources if target in transferRates[source].keys()])),
                    name="GlobiomConstraint2forTarget"+target)
        for source in sources:
            for target in transferRates[source].keys():
                if transferRates[source][target][1] == "secondary":
                    self.objectiveTypes["GlobiomSecondaryUsageof"+source+"as"+target] = [""]*5
                    self.objectiveTypes["GlobiomSecondaryUsageof"+source+"as"+target][0] = "Usage of "+source+" as "+target+" in Globiom demands"
                    self.objectiveTypes["GlobiomSecondaryUsageof"+source+"as"+target][2] = "min"
                    self.objective["GlobiomSecondaryUsageof"+source+"as"+target] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),"GlobiomSecondaryUsageof"+source+"as"+target)
                    self.maxDummyConstraints["GlobiomSecondaryUsageof"+source+"as"+target] = self.solver.Constraint(-self.solver.infinity(),self.solver.infinity(),"maxDummyConstraintforGlobiomTargetFor"+target)
                    self.solver.Add(self.objective["GlobiomSecondaryUsageof"+source+"as"+target] == sum(self.globiomProduction[source][year][target] for year in self.years),
                        name="GlobiomSecondaryUsageConstraintof"+source+"as"+target)

    def calculateObjectiveRanges(self,debug=False):
        self.debug=debug
        lb = {objName:np.inf for objName  in self.objectiveTypes.keys()}
        ub = {objName:-np.inf for objName  in self.objectiveTypes.keys()}
        
        display_local("Calculating objective ranges")
        #for i,objName in enumerate(tqdm(self.objectiveTypes.keys(),file=sys.stdout)):
        for i,objName in enumerate(self.objectiveTypes.keys()):
            self.objectiveFunction = self.solver.Objective()
            #display("Optimizing for "+self.objectiveTypes[objName][0])
            self.objectiveFunction.SetMaximization()
            if self.objectiveTypes[objName][2] == "max":
                multiplier = 1
            elif self.objectiveTypes[objName][2] == "min":
                multiplier = -1
            for objName2 in self.objectiveTypes.keys():
                if objName == objName2:
                    self.objectiveFunction.SetCoefficient(self.objective[objName2],multiplier)
                else:
                    # If ranges already calculated also add the other objectives with small coefficients to improve ranges:
                    try:
                        self.objectiveFunction.SetCoefficient(self.objective[objName2],multiplier*1e-6/(self.objectiveRanges[objName2][1]-self.objectiveRanges[objName2][0]))
                    except AttributeError:
                        self.objectiveFunction.SetCoefficient(self.objective[objName2],0)                        
            #If we have already been running the GUI, then we need to remove maxDummy from objective function
            try:
                self.objectiveFunction.SetCoefficient(self.maxDummy,0)
            except AttributeError:
                pass
            if self.debug:
                problem = self.solver.ExportModelAsLpFormat(obfuscated=False)
                print(problem,file=open("problem.lp","w"))
            now = datetime.now()
            res = self.solver.Solve()
            time = datetime.now() - now 
            self.solutionTimeStamp = str(now).replace(":"," ")
            if res == self.solver.OPTIMAL:
                #display("Found an optimal solution in "+str(time.seconds)+" seconds")
                #display("Objective values are:")
                for i,objName in enumerate(self.objectiveTypes.keys()):
                    #display(self.objectiveTypes[objName][0],self.objective[objName].solution_value())
                    if self.objective[objName].solution_value() > ub[objName]:
                        ub[objName] = self.objective[objName].solution_value()
                    if self.objective[objName].solution_value() < lb[objName]:
                        lb[objName] = self.objective[objName].solution_value()
            else:
                display("Ratkaisua ei löytynyt")
                if res == self.solver.FIXED_VALUE:
                    display("Objective value fixed")
                if res == self.solver.INFEASIBLE:
                    display("Ongelma ei ole ratkaistavissa")
                if res == self.solver.ABNORMAL:
                    display("Something strange in the problem")
                if res == self.solver.NOT_SOLVED:
                    display("Problem could not be solved for some reason")            
        self.objectiveRanges = {objName: (lb[objName],ub[objName]) for objName in self.objectiveTypes.keys()}           


    def defineEpsilonConstraint(self,**kwargs):
        try:
            for objName in self.objectiveTypes.keys():
                self.epsilonConstraints[objName].SetCoefficient(self.objective[objName],1)
        except AttributeError:
            self.epsilonConstraints = {
                objName:self.solver.Constraint(-self.solver.infinity(),self.solver.infinity(),"epsilonConstraintFor"+objName)
                for objName in self.objectiveTypes.keys()
            }
            for objName in self.objectiveTypes.keys():
                self.epsilonConstraints[objName].SetCoefficient(self.objective[objName],1)
        epsilonValues = kwargs
        for objName in epsilonValues.keys():
            if self.objectiveTypes[objName][2] == "max":
                self.epsilonConstraints[objName].SetLb(epsilonValues[objName])
            elif self.objectiveTypes[objName][2] == "min":
                self.epsilonConstraints[objName].SetUb(epsilonValues[objName])

    def selection_fn(trace,points,selector):
        #geodf2 = geodf1[(geodf1['year'] == kwargs['year']) & (geodf1['regime'] == kwargs['regime'])]
        #if 't' in locals():
        #    t.data[0].cells.values = [self.geodf1.iloc[points.point_inds][col] for col in ['id','Harvested_V','SC','Biomass']]#variables of interest in map -- could be a cross box (?)
        global sel_stand
        bb = [[self.geodf1.iloc[points.point_inds][col] for col in ['id']]]
        sel_stand = [bb[0][0].iloc[i] for i in range(0,len(bb[0][0]))]
        print(sel_stands)

    def on_display_ADD_CONST(self):
        if "SPATIAL" in self.constraints:
            for standId in self.sel_stand:
                self.constraints["SPATIAL"][standId] = self.solver.Add(self.regimesDecision[(standId,self.regime)]==1,name = "Only"+self.regime+"onStand"+str(standId))
        else:
            self.constraints["SPATIAL"] = dict()
            for standId in self.sel_stand:
                self.constraints["SPATIAL"][standId] = self.solver.Add(self.regimesDecision[(standId,self.regime)]==1,name = "Only"+self.regime+"onStand"+str(standId))
    
    def selection_fn_CONST(self,trace,points,selector):
        geodf3 = self.geodf2[(self.geodf2['year'] == 2116) & (self.geodf2['regime'] == self.regime)]
        
        bb = [[geodf3.iloc[points.point_inds][col] for col in ['id']]]
        self.sel_stand = [bb[0][0].iloc[i] for i in range(0,len(bb[0][0]))]
        colStandConst = widgets.interactive(self.on_display_ADD_CONST,{"manual":True,"manual_name": "Spatial Constraint" + str(len(self.sel_stand))+ " Stands as "+self.regime})
        display(colStandConst)
        
        
        
    def on_display_map_CONST(self,**kwargs):
        display(kwargs)
        clear_output()
        geodf3 = self.geodf2[(self.geodf2['year'] == 2116) & (self.geodf2['regime'] == kwargs['regime'])]
        fig = px.choropleth_mapbox(geodf3.set_index("id"),    geojson=geodf3.geometry,    locations=geodf3.index,    color="V",center=dict(lat=60.519712773049015, lon=27.521227544658096),    mapbox_style="open-street-map",opacity =0.4,    zoom=9,)
        fig.update_layout(    height=500,    autosize=True,    margin={"r": 0, "t": 0, "l": 0, "b": 0})#
        ff =go.FigureWidget(fig)
        scatter = ff.data[0]
        self.regime=kwargs['regime']
        scatter.on_selection(self.selection_fn_CONST)
        display(VBox([ff]))
        
    
    def removeSpatialConstraints(self):
        self.constraints["SPATIAL"] = dict()
    
    
    def defineReferencePointAndSolve(self,**kwargs):
        display_local("Starting problem solving at "+str(datetime.now()))
        self.solutionCounter +=1
        referencePoint = kwargs
        for objName in self.objectiveTypes.keys():
            self.maxDummyConstraints[objName].SetCoefficient(self.maxDummy,1)
            if self.objectiveTypes[objName][2] == "max":
                self.maxDummyConstraints[objName].SetCoefficient(self.objective[objName],-1/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]))
                self.maxDummyConstraints[objName].SetUb(-referencePoint[objName]/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]))
                self.objectiveFunction.SetCoefficient(self.objective[objName],10**(-6)/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]))
            elif self.objectiveTypes[objName][2] == "min":
                self.maxDummyConstraints[objName].SetCoefficient(self.objective[objName],1/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]))
                self.maxDummyConstraints[objName].SetUb(referencePoint[objName]/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]))
                self.objectiveFunction.SetCoefficient(self.objective[objName],-10**(-6)/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]))

        self.objectiveFunction.SetCoefficient(self.maxDummy,1)
        if self.debug:
            problem = self.solver.ExportModelAsLpFormat(obfuscated=False)
            print(problem,file=open("problem.lp","w"))
            display_local("Problem exported at "+str(datetime.now()))
        
        now = datetime.now()
        res = self.solver.Solve()
        time = datetime.now() - now
        display_local("Problem solved at "+str(datetime.now()))
        self.solutionTimeStamp = str(now).replace(":"," ")
        if res == self.solver.OPTIMAL:
            if self.LOC == "ANSSI":
                display(HTML("<h3>OUTCOMES</h3>"))
                display({self.objectiveTypes[objName][0]:self.objective[objName].solution_value() for objName in self.objectiveTypes.keys()})
            elif self.LOC == "HS":
                display(HTML("<h3>OUTCOMES</h3>"))
                display({self.objectiveTypes[objName][0]:self.objective[objName].solution_value() for objName in self.objectiveTypes.keys()})
            else:
                #for i,objName in enumerate(self.objectiveTypes.keys()):
                #    #Try to set the value of the slider if it exists
                #    try:
                #        self.imRef.children[i].value = self.objective[objName].solution_value()
                #    except:
                #        pass
                display_local("Found an optimal solution in "+str(time.seconds)+" seconds")
                display_local ({self.objectiveTypes[objName][0]:self.objective[objName].solution_value() for objName in self.objectiveTypes.keys()})
                display_local("Solution printed at "+str(datetime.now()))
                
                self.result = self.CalculateResults()
                display_local("Ready for Maps & Graphs at "+str(datetime.now()))
                layout = widgets.Layout(width='auto', height='40px') 
                self.printSolutionButton = widgets.Button(description='Print solution',layout=layout)
                self.printSolutionButton.on_click(self.printSolution)
                display_local(self.printSolutionButton)
                
                self.results_output =widgets.Output()
                with self.results_output:
                    display(HTML("<h2>Tulokset</h2>"))
                    self.show_regime()
                    display(HTML("<h3>Arvioitu kehityskulku</h3>"))
                    self.opt_GraphChooser = widgets.interactive(
                        self.create_line,
                        feature1=widgets.Dropdown(
                            options=[(self.unit_head.get(col) or col, col) for col in self.result.columns],
                            description="Mittari",
                        ),
                    )
                    display(self.opt_GraphChooser)
                    result = self.result
                    if self.spatial != "None":
                        display(HTML("<h3>Arvioitu jakauma</h3>"))
                        geodf = gpd.read_file(module_path + "data/compressed/"+self.spatial) ## Data with geometry ##EDIT
                        # shape file is a different CRS,  change to lon/lat GPS co-ordinates
                        geodf = geodf.to_crs("WGS84")

                        #Merging databases together - linking spatial information with simulated
                        geometry = pd.DataFrame(list(set(result.index.get_level_values(0))),geodf['geometry'])
                        geometry = geometry.reset_index()
                        d = {'geometry':geodf['geometry'], 'id': list(set(result.index.get_level_values(0)))}
                        geometry = pd.DataFrame(d)

                        self.geodf1 = result.reset_index().merge(geometry,right_on="id",left_on="id")
                        geometry = self.geodf1['geometry']
                        self.geodf1.drop('geometry',axis=1)
                        self.geodf1 = gpd.GeoDataFrame(self.geodf1,crs="WGS84",geometry=geometry)

                        opt_TypeChooser = widgets.interactive(
                            self.on_display_map_opt,
                            values=widgets.Dropdown(
                                options=[(self.unit_head.get(col) or col, col) for col in list(self.geodf1.columns)[2:-1]],
                                description="Mittari",
                            ),
                            Vuosi=set(self.geodf1['year'])
                        )
                        display(opt_TypeChooser)
                        
                        dat = pd.read_csv(self.filename)

                        #Merging databases together - linking spatial information with simulated
                        geometry = pd.DataFrame(list(set(dat['id'])),geodf['geometry'])
                        d = {'geometry':geodf['geometry'], 'id1': list(set(dat['id']))}
                        geometry = pd.DataFrame(d)

                        self.geodf2 = dat.merge(geometry,left_on="id",right_on="id1")
                        geometry = self.geodf2['geometry']
                        self.geodf2.drop('geometry',axis=1)
                        self.geodf2 = gpd.GeoDataFrame(self.geodf2,crs="WGS84",geometry=geometry)
                        return {objName:self.objective[objName].solution_value() for objName in self.objectiveTypes.keys()}
        else:
            display("Ratkaisua ei löytynyt")
            if res == self.solver.FIXED_VALUE:
                display("Objective value fixed")
            if res == self.solver.INFEASIBLE:
                display("Ongelma ei ole ratkaistavissa")
            if res == self.solver.ABNORMAL:
                display("Something abnormal in the problem")
            if res == self.solver.NOT_SOLVED:
                display("Problem could not be solved for some reason")      

    def enableAndDisableConstraints(self,**kwargs):
        enabledConstraints = kwargs
        for constraintName in self.constraints.keys():
            if self.constraintTypes[constraintName][0] == "SPATIAL":
                for standId in self.standIds:
                    if (standId) in self.constraints['SPATIAL'].keys():
                        for regime in self.regimes:
                            if (standId,regime) in self.regimesDecision.keys():
                                self.constraints[constraintName][standId].SetUb(0)
            if self.constraintTypes[constraintName][0] == "Allowed regimes":
                for regime in self.regimes:
                    if regime not in self.constraintTypes[constraintName][2]:
                        for standId in self.standIds:
                            if (standId,regime) in self.regimesDecision.keys() and self.data.loc[(standId,self.years[0],regime),self.constraintTypes[constraintName][3]] == 1:
                                self.constraints[constraintName][(standId,regime)].SetUb(1-(enabledConstraints[constraintName]))
            if self.constraintTypes[constraintName][0] == "Species reduction":
                periodNo = self.constraintTypes[constraintName][3]
                for i,year in enumerate(self.years):
                    if i >= periodNo:
                        self.constraints[constraintName][year].SetLb((-1+enabledConstraints[constraintName])*1e10)
            if self.constraintTypes[constraintName][0] == "less than": ## Added by JK from older version
                for year in self.years: ## Added by JK from older version
                    self.constraints[constraintName][year].SetUb((1-enabledConstraints[constraintName])*1e10)

    def printSolution(self):
        import os
        if not os.path.isdir("./results"):
            os.mkdir("./results")
        with open("./results/objectiveValues"+str(self.solutionCounter)+".csv","w") as file:
            delim = ""
            for objName in self.objectiveTypes.keys():
                file.write(delim+objName)
                delim = ","
            file.write("\n")
            delim = ""
            for objName in self.objectiveTypes.keys():
                file.write(delim+str(self.objective[objName].solution_value()))
                delim = ","
            file.write("\n")

    def CalculateResults(self):
        t4 = self.regimesDecision
        reformat = {}
        for k in t4.keys():
            reformat[k]=[t4[k]]
        t3 = pd.DataFrame(reformat)
        t3 =t3.transpose()
        if self.LOC == "GER":
            t6 = pd.DataFrame(self.data.loc[(slice(None),slice(None),slice(None)),list(self.unit.keys())])
            t7 = t6.reset_index()
            t7.set_index(['id','regime','year'], inplace=True)
            tt3 = t7[['HarvestedVolume']]
            tt3 = tt3.rename(columns={"HarvestedVolume":"Decision"})
            self.tt3=tt3
            for (s,r) in t3.index:
                tt3.loc[(s,r,slice(None)),"Decision"] = t3.loc[(s,r)][0].solution_value()
            tt7 = t7.reset_index().set_index(['id','regime','year'])
            result=tt7.mul(tt3['Decision'],axis=0)#.max()['V']    
            result = result.groupby(["id","year"]).sum()
            self.result = result
        if self.LOC == "FINN":
            t6 = pd.DataFrame(self.data.loc[(slice(None),slice(None),slice(None)),['V', 'i_Vm3', 'Harvested_V', 'Harvested_V_log_under_bark',
                   'Harvested_V_pulp_under_bark', 'Harvested_V_under_bark', 'MAIN_SP',
                   'Age', 'AGE_ba', 'SC', 'Biomass', 'ALL_MARKETED_MUSHROOMS', 'BILBERRY',
                   'COWBERRY', 'HSI_MOOSE', 'CAPERCAILLIE', 'HAZEL_GROUSE',
                   'V_total_deadwood', 'N_where_D_gt_40', 'PEAT','CARBON_SINK', 'prc_V_deciduous',
                   'clearcut', 'CARBON_STORAGE_Update', 'Recreation','Disc_cash_flow_3','Disc_cash_flow_1',
                   'Scenic',]])#
            t7 = t6.reset_index()
            t7.set_index(['id','regime','year'], inplace=True)
            tt3 = t7[['V']]
            tt3 = tt3.rename(columns={"V":"Decision"})
            self.tt3=tt3
            for (s,r) in t3.index:
                tt3.loc[(s,r,slice(None)),"Decision"] = t3.loc[(s,r)][0].solution_value()
            tt7 = t7.reset_index().set_index(['id','regime','year'])
            result=tt7.mul(tt3['Decision'],axis=0)
            result = result.groupby(["id","year"]).sum()
            self.result = result
            self.AREA_REGIMES = (self.data.loc[slice(None),slice(None),slice(None)]['AREA']*tt3['Decision']).to_frame().loc[slice(None),self.data.index.get_level_values(1)[0],slice(None)].groupby("regime").sum()
            
        if self.LOC == "ANSSI":
            t6 = pd.DataFrame(self.data.loc[(slice(None),slice(None),slice(None)),['BIOMASS', 'DEADWOOD', 'NPV3',]])#
            t7 = t6.reset_index()
            t7.set_index(['id','regime','year'], inplace=True)
            tt3 = t7[['BIOMASS']]
            tt3 = tt3.rename(columns={"V":"Decision"})
            self.tt3=tt3
            for (s,r) in t3.index:
                tt3.loc[(s,r,slice(None)),"Decision"] = t3.loc[(s,r)][0].solution_value()
            tt7 = t7.reset_index().set_index(['id','regime','year'])
            result=tt7.mul(tt3['Decision'],axis=0)
            result = result.groupby(["id","year"]).sum()
            self.result = result
            self.AREA_REGIMES = (self.data.loc[slice(None),slice(None),slice(None)]['AREA']*tt3['Decision']).to_frame().loc[slice(None),self.data.index.get_level_values(1)[0],slice(None)].groupby("regime").sum()
        if self.LOC == "HS":
            t6 = pd.DataFrame(self.data.loc[(slice(None),slice(None),slice(None)),list(self.VARS.keys())])#
            t7 = t6.reset_index()
            t7.set_index(['id','regime','year'], inplace=True)
            tt3 = t7[[list(self.VARS.keys())[0]]]
            tt3 = tt3.rename(columns={"V":"Decision"})
            self.tt3=tt3
            for (s,r) in t3.index:
                tt3.loc[(s,r,slice(None)),"Decision"] = t3.loc[(s,r)][0].solution_value()
            tt7 = t7.reset_index().set_index(['id','regime','year'])
            result=tt7.mul(tt3['Decision'],axis=0)
            result = result.groupby(["id","year"]).sum()
            self.result = result
            self.AREA_REGIMES = (self.data.loc[slice(None),slice(None),slice(None)]['AREA']*tt3['Decision']).to_frame().loc[slice(None),self.data.index.get_level_values(1)[0],slice(None)].groupby("regime").sum()
        if self.LOC == "NOR":
            t6 = pd.DataFrame(self.data.loc[(slice(None),slice(None),slice(None)),list(self.unit.keys())])
            t7 = t6.reset_index()
            t7.set_index(['plot_id','regime','year'], inplace=True)
            tt3 = t7[['harv_net_Mnok_TotNor']]
            tt3 = tt3.rename(columns={"harv_net_Mnok_TotNor":"Decision"})
            self.tt3=tt3
            for (s,r) in t3.index:
                tt3.loc[(s,r,slice(None)),"Decision"] = t3.loc[(s,r)][0].solution_value()
            tt7 = t7.reset_index().set_index(['plot_id','regime','year'])
            result=tt7.mul(tt3['Decision'],axis=0)
            result = result.groupby(["plot_id","year"]).sum()
            self.result = result
        if self.LOC =="SWE":
            t6 = pd.DataFrame(self.data.loc[(slice(None),slice(None),slice(None)),['Age','StandingVolume', 'VolumeDecidous', 'SumVolumeCutTotal','SumTimberVolumeTotal', 'SumPulpVolumeTotal', 'SumHarvestResiduesTotal','SumHarvestFuelwoodTotal', 'AnnualIncrementNetTotal', 'DeadWoodVolume','reserve', 'NPV', 'RecreationIndex', 'TotalCarbon', 'PulpFuel','SimulatedSAWlog', 'SimulatedResidue', 'SimulatedPulPFuel','DeciduousRatio', 'old_deciduous_rich_forest_area', 'SetAside','managed', '110%_of_periodic_increment_managed','90%_of_periodic_increment_managed', 'Total_VolumeDeciduous','Total_DeadWoodVolume', 'Total_RecreationIndex', 'Total_TotalCarbon','Relative_RepresentedArea', 'Relative_Age', 'Relative_StandingVolume','Relative_VolumeDecidous', 'Relative_SumVolumeCutTotal','Relative_SumTimberVolumeTotal', 'Relative_SumPulpVolumeTotal','Relative_SumHarvestResiduesTotal', 'Relative_SumHarvestFuelwoodTotal','Relative_DeadWoodVolume', 'Relative_RecreationIndex','Relative_TotalCarbon', 'Relative_PulpFuel', 'Relative_SimulatedSAWlog','Relative_SimulatedResidue', 'Relative_SimulatedPulPFuel','Relative_DeciduousRatio', 'Relative_old_deciduous_rich_forest_area','Relative_Total_VolumeDeciduous', 'Relative_Total_DeadWoodVolume','Relative_Total_RecreationIndex', 'Relative_Total_TotalCarbon',]])
            t7 = t6.reset_index()
            t7.set_index(['Description','period','combinedRegime'], inplace=True)
            tt3 = t7[['StandingVolume']]
            tt3 = tt3.rename(columns={"StandingVolume":"Decision"})
            self.tt3=tt3
            for (s,r) in t3.index:
                tt3.loc[(s,slice(None),r),"Decision"] = t3.loc[(s,r)][0].solution_value()
            tt7 = t7.reset_index().set_index(['Description','period','combinedRegime'])
            result=tt7.mul(tt3['Decision'],axis=0)#.max()['V']    
            result = result.groupby(['Description','period']).sum()
            self.result = result
            
        return result
    
    def to_print_output(self):
        import os
        from datetime import datetime
        n = datetime.now()
        dt_string = n.strftime("%H%M_%S")

        if not os.path.isdir("./results"):
            os.mkdir("./results")
        print_geodf2 = self.geodf2.drop(["geometry"],axis =1)
        print_geodf2.to_csv("./results/Stand_level_results_at_"+dt_string+".csv")
    
    def on_display_map_opt(self,**kwargs):
        
        display(kwargs)
        clear_output()
        self.geodf2b = self.geodf1[(self.geodf1['year'] == kwargs['Vuosi']) & (self.geodf1[kwargs['values']] >=0.001) ]
        self.geodf2b.set_index("id")
        fig = px.choropleth_mapbox(self.geodf2b,    geojson=self.geodf2b.geometry,    locations=self.geodf2b.index,    color=kwargs['values'],    center=dict(lat=60.519712773049015, lon=27.521227544658096),    mapbox_style="open-street-map",opacity =0.4,    zoom=11,color_continuous_scale=[[0, 'rgb(240,240,240)'],
                          [0, 'rgb(4,145,32)'],
                          [1, 'rgb(227,26,28,0.5)']])
        fig.update_layout(coloraxis_colorbar=dict(title=kwargs['values']+" "+self.unit[kwargs['values']]),    height=500,    autosize=True,    margin={"r": 0, "t": 0, "l": 0, "b": 0})#,    paper_bgcolor="#303030",    plot_bgcolor="#303030",)
        ff =go.FigureWidget(fig)
        scatter = ff.data[0]
        scatter.on_selection(self.selection_fn)
        display(VBox([ff]))
        colOutputPrint = widgets.interactive(self.to_print_output,{"manual":True,"manual_name": "Write complete solution to file"})
        display_local(colOutputPrint)

    def create_line(self,**kwargs):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        if self.LOC =="FINN":
            area = self.data.loc[slice(None),2024,"SA"]['AREA'].sum()
            title_Y = self.unit_head[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}")
            (self.result.groupby(['year']).sum()/area).plot(use_index=True,y=kwargs['feature1'],legend=False,title=title_Y,ylabel=title_Y+" "+self.unit[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}"),xlabel="Vuosi",ax=ax)
        if self.LOC =="ANSSI":
            area = self.data.loc[slice(None),2050,"initial_regime"]['AREA'].sum()
            title_Y = self.unit_head[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}")
            (self.result.groupby(['year']).sum()/area).plot(use_index=True,y=kwargs['feature1'],legend=False,title=title_Y,ylabel=title_Y+" "+self.unit[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}"),xlabel="Vuosi",ax=ax)
        if self.LOC =="HS":
            area = self.data.loc[slice(None),2050,"initial_regime"]['AREA'].sum()
            title_Y = self.unit_head[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}")
            (self.result.groupby(['year']).sum()/area).plot(use_index=True,y=kwargs['feature1'],legend=False,title=title_Y,ylabel=title_Y+" "+self.unit[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}"),xlabel="Vuosi",ax=ax)
        if self.LOC =="GER":
            area = self.data.loc[slice(None),2017,"NOT"]['represented_area_by_NFIplot'].sum()
            title_Y = self.unit_head[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}")
            (self.result.groupby(['year']).sum()/area).plot(use_index=True,y=kwargs['feature1'],legend=False,title=title_Y,ylabel=title_Y+" "+self.unit[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}"),xlabel="Vuosi",ax=ax)
        if self.LOC == "NOR":
            area = self.data.loc[slice(None),2028,"SimOpt_no_management_0"]['tsd_ha2total'].sum()
            title_Y = self.unit_head[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}")
            (self.result.groupby(['year']).sum()/area).plot(use_index=True,y=kwargs['feature1'],legend=False,title=title_Y,ylabel=title_Y+" "+self.unit[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}"),xlabel="Vuosi",ax=ax)
        if self.LOC == "SWE":
            #area = self.data.loc[slice(None),2028,"SimOpt_no_management_0"]['tsd_ha2total'].sum()
            (self.result.groupby(['period']).sum()).plot(use_index=True,y=kwargs['feature1'],legend=False,title=kwargs['feature1'],ylabel=kwargs['feature1'],xlabel="Year",ax=ax)
            #title_Y = self.unit_head[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}")
            #(self.result.groupby(['year']).sum()).plot(use_index=True,y=kwargs['feature1'],legend=False,title=title_Y,ylabel=title_Y+" "+self.unit[kwargs['feature1']].replace("<sup>","$^").replace("</sup>","$").replace("<sub>","$_").replace("</sub>","$").replace("-1","{-1}"),xlabel="Year",ax=ax)
        plt.show()
        
    def show_regime(self):
        if self.LOC =="FINN":
            REGS = (self.data.loc[slice(None),slice(None),slice(None)]['AREA']*self.tt3['Decision']).to_frame().loc[slice(None),2024,slice(None)].groupby("regime").sum()
            dic_BAU = {'I-Jakso': ["BAU_m5","BAUwT_m5","BAUwoT_m20","BAU_F","BAUwT_F","BAU_m5_F","BAUwT_m5_F","BAUwo5_m20_F"],
                       'Jakso':["BAU","BAUwT","BAUwoT"],
                       'Jakso-ilmasto':["BAUwT_B","BAUwT_5_B","BAUwT_15_B","BAUwT_30_B",'BAU_GTR_B'],#"BAUwT_GTR_B",
                       'E-Jakso':["BAUwT_GTR","BAUwGTR","BAU_5","BAUwT_5","BAU_15","BAUwT_15","BAU_30","BAUwT_30"],
                       'Jatkuva':["CCF_1","CCF_2","CCF_3","CCF_4"],
                       'Suojelu':["SA"]}
            dd = {i:(REGS.loc[dic_BAU[i]].sum()[0]/sum(REGS[0]))*100 for i in dic_BAU.keys()}
        elif self.LOC =="ANSSI":
            REGS = (self.data.loc[slice(None),slice(None),slice(None)]['AREA']*self.tt3['Decision']).to_frame().loc[slice(None),2050,slice(None)].groupby("regime").sum()
            dic_BAU = {'I-Jakso': [i for i in range(1,80)],#["BAU_m5","BAUwT_m5","BAUwoT_m20","BAU_F","BAUwT_F","BAU_m5_F","BAUwT_m5_F","BAUwo5_m20_F"],
                       'Jakso':["BAU","BAUwT","BAUwoT"],
                       'Jakso-ilmasto':["BAUwT_B","BAUwT_5_B","BAUwT_15_B","BAUwT_30_B",'BAU_GTR_B'],#"BAUwT_GTR_B",
                       'E-Jakso':["BAUwT_GTR","BAUwGTR","BAU_5","BAUwT_5","BAU_15","BAUwT_15","BAU_30","BAUwT_30"],
                       'Jatkuva':["CCF_1","CCF_2","CCF_3","CCF_4"],
                       'Suojelu':["SA"]}
            dd = {i:(REGS.loc[dic_BAU[i]].sum()[0]/sum(REGS[0]))*100 for i in dic_BAU.keys()}
        elif self.LOC =="HS":
            REGS = (self.data.loc[slice(None),slice(None),slice(None)]['AREA']*self.tt3['Decision']).to_frame().loc[slice(None),2050,slice(None)].groupby("regime").sum()
            dic_BAU = {'I-Jakso': ['BAU', 'BAUnDIACUT', 'BAUpDIACUT', 'BAUpTHININT','BAUnTHININT', 'BAUppRESIDUAL', 'BAUpRESIDUAL', 'BAUpN'],#[i for i in range(1,80)],#["BAU_m5","BAUwT_m5","BAUwoT_m20","BAU_F","BAUwT_F","BAU_m5_F","BAUwT_m5_F","BAUwo5_m20_F"],
                       'Jakso':["BAU","BAUwT","BAUwoT"],
                       'Jakso-ilmasto':["BAUwT_B","BAUwT_5_B","BAUwT_15_B","BAUwT_30_B",'BAU_GTR_B'],#"BAUwT_GTR_B",
                       'E-Jakso':["BAUwT_GTR","BAUwGTR","BAU_5","BAUwT_5","BAU_15","BAUwT_15","BAU_30","BAUwT_30"],
                       'Jatkuva':["CCF_1","CCF_2","CCF_3","CCF_4"],
                       'Suojelu':["SA"]}
            dd = {i:(REGS.loc[dic_BAU[i]].sum()[0]/sum(REGS[0]))*100 for i in dic_BAU.keys()}
        elif self.LOC =="GER":
            REGS = (self.data.loc[slice(None),slice(None),slice(None)]['Relative_represented_area_by_NFIplot']*self.tt3['Decision']).to_frame().loc[slice(None),2017,slice(None)].groupby("regime").sum()
            dic_BAU = {
                       'IBAU': ["BAU_FS1","BAU_RR","BAU_RR_p2","BAU_RR_p1"],
                       'BAU':["BAU_0","BAU_0_p2","BAU_0_p1"],
                       'EBAU':["BAU_1"],
                       'CCF':["CCF_P2","CCF_P3","CCF_P3_p1","CCF_P3_p2","CCF_P1"],
                       'ACC':["CCF_STATE"],
                       'SA':["NOT"]}
            dd = {i:(REGS.loc[dic_BAU[i]].sum()[0]/sum(REGS[0]))*100 for i in dic_BAU.keys()}
        elif self.LOC == "NOR":
            REGS = (self.data.loc[slice(None),slice(None),slice(None)]['tsd_ha2total']*self.tt3['Decision']).to_frame().loc[slice(None),2028,slice(None)].groupby("regime").sum()
            dic_BAU = {'AAC': ["SimOpt_multispecies_0","SimOpt_multispecies_5","SimOpt_multispecies_1","SimOpt_multispecies_4","SimOpt_multispecies_15","SimOpt_multispecies_13","SimOpt_multispecies_3","SimOpt_multispecies_2","SimOpt_multispecies_18","SimOpt_multispecies_10","SimOpt_multispecies_11","SimOpt_multispecies_12","SimOpt_multispecies_14","SimOpt_multispecies_16","SimOpt_multispecies_7","SimOpt_multispecies_17","SimOpt_multispecies_9","SimOpt_multispecies_8","SimOpt_multispecies_6"],
               "BAU": ["SimOpt_extensive_0","SimOpt_extensive_18","SimOpt_extensive_13","SimOpt_extensive_9","SimOpt_extensive_17","SimOpt_extensive_2","SimOpt_extensive_6","SimOpt_extensive_1","SimOpt_extensive_8","SimOpt_extensive_14","SimOpt_extensive_15","SimOpt_extensive_10","SimOpt_extensive_16","SimOpt_extensive_4","SimOpt_extensive_11","SimOpt_extensive_5","SimOpt_extensive_3","SimOpt_extensive_7","SimOpt_extensive_12"],
               "CCF": ["SimOpt_ccover_0","SimOpt_ccover_1","SimOpt_ccover_2"],
               "E-BAU": ["SimOpt_extensive_long_0","SimOpt_extensive_long_10","SimOpt_extensive_long_18","SimOpt_extensive_long_1","SimOpt_extensive_long_11","SimOpt_extensive_long_9","SimOpt_extensive_long_13","SimOpt_extensive_long_6","SimOpt_extensive_long_15","SimOpt_extensive_long_12","SimOpt_extensive_long_5","SimOpt_extensive_long_4","SimOpt_extensive_long_8","SimOpt_extensive_long_16","SimOpt_extensive_long_14","SimOpt_extensive_long_3","SimOpt_extensive_long_17","SimOpt_extensive_long_7","SimOpt_extensive_long_2"],
               "I-BAU": ["SimOpt_int_0","SimOpt_int_short_0","SimOpt_int_short_1","SimOpt_int_short_4","SimOpt_int_3","SimOpt_int_12","SimOpt_int_2","SimOpt_int_8","SimOpt_int_short_2","SimOpt_int_4","SimOpt_int_short_10","SimOpt_int_18","SimOpt_int_7","SimOpt_int_1","SimOpt_int_short_17","SimOpt_int_short_8","SimOpt_int_13","SimOpt_int_9","SimOpt_int_10","SimOpt_int_14","SimOpt_int_short_13","SimOpt_int_short_9","SimOpt_int_short_5","SimOpt_int_short_12","SimOpt_int_short_7","SimOpt_int_short_11","SimOpt_int_11","SimOpt_int_short_16","SimOpt_int_short_6","SimOpt_int_short_14","SimOpt_int_6","SimOpt_int_short_3","SimOpt_int_5","SimOpt_int_15","SimOpt_int_short_15","SimOpt_int_short_18","SimOpt_int_16","SimOpt_int_17"],
               "SA": ["SimOpt_no_management_0"]}
            dd = {i:(REGS.loc[dic_BAU[i]].sum()[0]/sum(REGS[0]))*100 for i in dic_BAU.keys()}
        elif self.LOC == "SWE":
            data_slice = self.data.loc[slice(None), slice(None), slice(None)]
            REGS = (data_slice['RepresentedArea'] * self.tt3['Decision']).to_frame().loc[slice(None), 1, slice(None)]
            REGS1 = REGS.reset_index()
            REGS1[['combinedRegime']] = REGS1['combinedRegime'].apply(lambda x: pd.Series(str(x).split("_")))[[0]]
            REGS1['combinedRegime'] = REGS1['combinedRegime'].str[19:]
            REGS1 = REGS1.set_index(['Description', 'combinedRegime']).groupby("combinedRegime").sum()
            # Define management regimes dictionary and compute dd
            dic_BAU = {'AAC': ["Lovgynnande trakthyggesbruk"],
                       "BAU": ["BAU"],
                       "CCF": ["CCF"],
                       "E-BAU": ["BAU - NoThinning", "BAU_ProlongedRotation"],
                       "I-BAU": ["Int_HybridExotic", "Int_Contorta", "BAU FocusBioenergy", "BAU_FocusBioenergy_StumpHarvest", "Int_Prod"],
                       "SA": ["SetAside (Unmanaged)"]}
            dd = {}
            for i in dic_BAU.keys():
                k = 0
                for ii in dic_BAU[i]:
                    try:
                        dd[i] = dd.get(i, 0) + (REGS1.loc[ii][0] / sum(REGS[0])) * 100
                        k = 1
                    except:
                        pass

        fig = plt.figure(figsize=[12, 10], dpi=100)
        ax1, ax2 = plt.subplot2grid(shape=(4,4),loc=(1,2),rowspan=3,colspan=2,fig=fig), plt.subplot2grid(shape=(4,4),loc=(0,0),rowspan=1,colspan=4,fig=fig)
        pd.DataFrame({self.objectiveTypes[objName][0]: (self.objective[objName].solution_value() - self.objectiveRanges[objName][0]) / (self.objectiveRanges[objName][1] - self.objectiveRanges[objName][0]) * 100 for objName in reversed(self.objectiveTypes.keys())}, index=["VAL"]).T.plot.barh(ax=ax1, legend=False)
        ax1.set_title("Tavoitteen saavutus")
        ax1.set_xlabel("%")
        ax1.set_xlim(0, 100)
        pd.DataFrame(dd, index=[""]).plot.barh(stacked=True, ax=ax2, width=3)
        ax2.set_title("Metsänkäsittelymenetelmien osuudet koko pinta-alasta")
        ax2.set_xticks([i * 10 for i in range(0, 11)])
        ax2.legend(bbox_to_anchor=(0.5, -1.2), loc="lower center", ncol=len(dic_BAU.keys()))
        ax2.set_xlabel("Metsänkäsittelymenetelmä [%]")
        ax2.set_xlim([0, 100])
        plt.subplots_adjust(hspace=2)
        plt.show()
    
    def on_display_graph_opt1(self,_):
        #self.result = self.CalculateResults()
        
        opt_GraphChooser = widgets.interactive(self.create_line,{"manual":True,"manual_name": "Display / Update Graph"}, **{"feature1":self.result.columns})
        display(opt_GraphChooser)

    def selection_fn(self,trace,points,selector):
        bb = [[self.geodf1.iloc[points.point_inds][col] for col in ['id']]]
        self.sel_stand = [bb[0][0].iloc[i] for i in range(0,len(bb[0][0]))]

    def on_display_map(self,**kwargs):
        display(kwargs)
        clear_output()
        geodf2 = self.geodf1[(self.geodf1['year'] == kwargs['year']) & (self.geodf1['regime'] == kwargs['regime'])]
        fig = px.choropleth_mapbox(self.geodf2.set_index("id"),    geojson=self.geodf2.geometry,    locations=self.geodf2.index,    color=kwargs['values'],      mapbox_style="open-street-map",opacity =0.4,    zoom=9,)
        fig.update_layout(    height=500,    autosize=True,    margin={"r": 0, "t": 0, "l": 0, "b": 0})#
        ff =go.FigureWidget(fig)
        scatter = ff.data[0]
        scatter.on_selection(self.selection_fn)
        display(VBox([ff]))

    
    def showGUI(self,debug=False):
        self.debug=debug
        self.imEps = widgets.interactive(self.defineEpsilonConstraint,
                 **{objName:self.objectiveRanges[objName] for objName in self.objectiveTypes.keys()}
                )
        self.imRef = widgets.interactive(self.defineReferencePointAndSolve, {'manual': True}, 
                **{objName:self.objectiveRanges[objName] for objName in self.objectiveTypes.keys()}
                )
        
        self.imConst = widgets.interactive(self.enableAndDisableConstraints,
                        **{constraintName:False for constraintName in self.constraints.keys()})
        for i,constraintName in enumerate(self.constraints.keys()):
            
            self.imConst.children[i].description = self.constraintTypes[constraintName][1]
            self.imConst.children[i].layout.width = "auto"
            #self.imConst.children[i].layout.display = "flex"
            #self.imConst.children[i].layout.flex_flow = "column"
        '''
        for i,objName in enumerate(self.objectiveTypes.keys()):
            # automatically update constraints on mouse up, not every time the slider value changes
            actual_min = self.objectiveRanges[objName][0]
            actual_max = self.objectiveRanges[objName][1]
            range_size = actual_max - actual_min
            
            # Transformations for scaling
            def scale_to_percent(value):
                return 100 * (value - actual_min) / range_size
            
            def scale_from_percent(percent_value):
                return actual_min + (percent_value / 100) * range_size
            
            
            self.imEps.children[i].continuous_update = False
            self.imEps.children[i].layout.width = "95%"
            self.imEps.children[i].style.description_width = "60%"
            
            # FOR HOVERING TEXT:
            #print(self.objectiveTypes[objName][-1])
            #self.imEps.children[i].tooltip = self.objectiveTypes[objName][-1]#"TEST"
            #self.imRef.children[i].tooltip = self.objectiveTypes[objName][-1]#"TEST"
            
            self.imEps.children[i].description = self.objectiveTypes[objName][0]
            
            if self.objectiveTypes[objName][2] == "max":
                self.imEps.children[i].value = self.objectiveRanges[objName][0]
            elif self.objectiveTypes[objName][2] == "min":
                self.imEps.children[i].value = self.objectiveRanges[objName][1]
            
            # Update sliders for epsilon constraints
            self.imEps.children[i].min = 0
            self.imEps.children[i].max = 100
            self.imEps.children[i].value = 0#scale_to_percent(self.imEps.children[i].value)
            #print(self.objectiveTypes[objName][2],actual_min,actual_max,range_size,scale_to_percent(self.imEps.children[i].value),self.imEps.children[i].value)
            self.imEps.children[i].step = 1  # Step in percentage
            #self.imEps.children[i].readout_format = ".0f%%"  # Show percentage
            
            # Update sliders for reference points
            self.imRef.children[i].min = 0
            self.imRef.children[i].max = 100
            self.imRef.children[i].value = 50#scale_to_percent(self.imRef.children[i].value)
            self.imRef.children[i].step = 1  # Step in percentage
            
            range_size = self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]
            self.imRef.children[i].layout.width = "95%"
            self.imRef.children[i].style.description_width = "60%"
            self.imRef.children[i].description = self.objectiveTypes[objName][0]
            
            #self.imEps.children[i].step=0.01 * range_size
            #self.imRef.children[i].step=0.01 * range_size
            # display just enough decimals to show a difference whenever the slider is moved
            decimals = max(0, math.floor(-math.log10(self.imEps.children[i].step)))
            fmt = f".{decimals}f"
            self.imEps.children[i].readout_format = fmt
            self.imRef.children[i].readout_format = fmt
            
            # Add observers to map percentage back to actual values
            def update_eps_value(change, slider=self.imEps.children[i], obj=objName):
                slider.unobserve_all()  # Prevent infinite recursion
                slider.value = scale_to_percent(change["new"])
                slider.observe(lambda c: update_eps_value(c), names="value")
            
            def update_ref_value(change, slider=self.imRef.children[i], obj=objName):
                slider.unobserve_all()
                slider.value = scale_to_percent(change["new"])
                slider.observe(lambda c: update_ref_value(c), names="value")
            
            self.imEps.children[i].observe(update_eps_value, names="value")
            self.imRef.children[i].observe(update_ref_value, names="value")
        '''
        
        for i,objName in enumerate(self.objectiveTypes.keys()):
            self.imEps.children[i].layout.width = "95%"
            self.imEps.children[i].style.description_width = "60%"
            self.imEps.children[i].description = self.objectiveTypes[objName][0]
            if self.objectiveTypes[objName][2] == "max":
                self.imEps.children[i].value = self.objectiveRanges[objName][0]
            elif self.objectiveTypes[objName][2] == "min":
                self.imEps.children[i].value = self.objectiveRanges[objName][1]
            self.imRef.children[i].layout.width = "95%"
            self.imRef.children[i].style.description_width = "60%"
            self.imRef.children[i].description = self.objectiveTypes[objName][0]
            self.imEps.children[i].step=0.01*(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0])
            self.imRef.children[i].step=0.01*(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0])
            decimals = max(0, math.floor(-math.log10(self.imRef.children[i].step)))
            fmt = f".{decimals}f"
            self.imEps.children[i].readout_format =fmt
            self.imRef.children[i].readout_format =fmt
            
        self.imRef.children[-2].description = "Optimize"
        
        optimize_button = self.imRef.children[-2]
        
        #Remove optimization button, so that it can be placed at the bottom.
        imRef_container = widgets.VBox([child for i, child in enumerate(self.imRef.children) if i != len(self.imRef.children) - 2])
        
        if self.spatial != "None":
            self.geodf = self.geodf = gpd.read_file(module_path+"data/compressed/"+self.spatial)
            self.geodf = self.geodf.to_crs("WGS84")
            self.fig = px.choropleth_mapbox(    self.geodf.set_index("standid"),    geojson=self.geodf.geometry,    locations=self.geodf.index,    center=dict(lat= 61.72557, lon=23.20108), color="fertilityc",        mapbox_style="open-street-map",#"open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner" or "stamen-watercolor" center=
                                    opacity =0.4,    zoom=9,)
            self.fig.update_layout(    height=500,    autosize=True,    margin={"r": 0, "t": 0, "l": 0, "b": 0},    paper_bgcolor="#303030",    plot_bgcolor="#303030",)
        layout = widgets.Layout(width='auto', height='40px') 
        
        #Create a checkbox to toggle imEps display
        toggle_imEps = widgets.Checkbox(
            value=False,  # Default to showing the widgets
            description="Show Constraint Values",
            tooltip="Check to display the constraint value sliders"
        )
        #imEps_container = widgets.Output()
        output_container = widgets.Output()
        # Function to update display based on checkbox
        def toggle_imEps_display(change):
            output_container.clear_output()
            if change["new"]:  # If checkbox is checked
                with output_container:
                    display(ui_const)
            else:
                with output_container:
                    display("Constraints hidden")

        toggle_imEps.observe(toggle_imEps_display, names="value")
        output_layout = widgets.Layout(
            width='100%',        # Adjust the width as per your needs
            height='auto',     # You can also adjust the height to fit the content better
            overflow='auto',    # Enable scrolling if content exceeds the container
            #margin='20px auto', # Center the widgets horizontally with margins
            padding='10px'      # Add some padding for better visual spacing
        )
        results_placeholder = widgets.Output()

        # Define a callback for the "OPTIMIZE" button
        def on_optimize_clicked(b):
            # Call defineReferencePointAndSolve and update the results
            #results = self.defineReferencePointAndSolve(**{key: self.imRef.children[i].value for i, key in enumerate(self.objectiveTypes.keys())})
            with results_placeholder:
                results_placeholder.clear_output()  # Clear previous results
                if self.LOC == "ANSSI":
                    print("DONE")
                if self.LOC == "HS":
                    def create_map():
                        import matplotlib.pyplot as plt
                        import matplotlib.cm as cm
                        import matplotlib.colors as mcolors

                        # Step 1: Solution values
                        decision_values = [var.solution_value() for var in self.decisionFrame["Decision"]]

                        # Step 2: Prepare data
                        df_solution = self.decisionFrame.copy().reset_index()
                        df_solution["Value"] = decision_values
                        df_solution[['lat', 'lon']] = df_solution['level_0'].str.split('_', expand=True)
                        df_solution['lat'] = df_solution['lat'].astype(float)
                        df_solution['lon'] = df_solution['lon'].astype(float)

                        # Step 3: Filter active decisions
                        data_2d = df_solution[df_solution['Value'] > 0].copy()
                        
                        unique_types = sorted(data_2d['level_1'].unique())
                        
                        '''
                        # Step 4: Map unique management types to consistent colors using tab10
                        unique_types = sorted(data_2d['level_1'].unique())
                        cmap = cm.get_cmap('tab10')
                        color_map = {level: cmap(i % 10) for i, level in enumerate(unique_types)}  # Safe for up to 10 types

                        # Step 5: Assign color values for the scatter plot
                        data_2d['color'] = data_2d['level_1'].map(color_map)

                        # ===== Scatter Plot =====
                        plt.figure(figsize=(10, 6))
                        plt.scatter(
                            data_2d['lon'], data_2d['lat'],
                            c=data_2d['color'], s=100, edgecolor='black'
                        )

                        # Custom legend
                        handles = [
                            plt.Line2D([0], [0], marker='o', color='w', label=label,
                                       markerfacecolor=color_map[label], markersize=7)
                            for label in unique_types
                        ]
                        plt.legend(handles=handles, title='Management', bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.xlabel("Longitude")
                        plt.ylabel("Latitude")
                        plt.title("Forest management")
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()'''
                        import pandas as pd
                        import numpy as np
                        import xarray as xr

                        df = data_2d
                        df['treatment_code'] = df['level_1'].astype('category').cat.codes + 1
                        # Assuming df is your DataFrame with columns 'lat', 'lon', and 'level_1'

                        # 1. Get sorted unique lat/lon values
                        lats = sorted(df['lat'].unique())
                        lons = sorted(df['lon'].unique())

                        # 2. Create empty 2D array with dtype=object (or str if you're storing text)
                        data = np.full((len(lats), len(lons)), 0, dtype=object)

                        # 3. Create mappings from lat/lon to grid indices
                        lat_to_idx = {lat: i for i, lat in enumerate(lats)}
                        lon_to_idx = {lon: j for j, lon in enumerate(lons)}

                        # 4. Fill array with level_1 values
                        for _, row in df.iterrows():
                            i = lat_to_idx[row['lat']]
                            j = lon_to_idx[row['lon']]
                            data[i, j] = row['treatment_code']

                        # 5. Create DataArray
                        da = xr.DataArray(
                            data,
                            coords={'lat': lats, 'lon': lons},
                            dims=('lat', 'lon'),
                            name='treatment_decision'  # optional name
                        )

                        # Now `da` is a 2D DataArray with labeled lat/lon axes
                        import matplotlib.pyplot as plt
                        import matplotlib.colors as mcolors


                        treatment_map = dict(enumerate(df['level_1'].astype('category').cat.categories, start=1))
                        treatment_map[0] = "SA"



                        d2_data = da.values
                        d2_data = da.values.astype(int)
                        masked_data = np.where(d2_data == 0, np.nan, d2_data)
                        used_ids = np.unique(d2_data)

                        max_id = used_ids.max()
                        ncat = max(max_id, 1)

                        cmap = plt.get_cmap("Spectral", ncat)
                        cmap.set_bad('white')
                        boundaries = np.arange(0.5, ncat + 1.5, 1)
                        norm = mcolors.BoundaryNorm(boundaries, ncat)
                        
                        color_map = {treatment_map[i]: cmap(norm(i)) for i in range(1, ncat + 1)}
                        

                        fig, ax = plt.subplots(figsize=(8, 6))
                        lats = da.coords["lat"].values
                        lons = da.coords["lon"].values

                        im = ax.pcolormesh(lons, lats, masked_data, cmap=cmap, norm=norm, shading='auto')
                        ax.set_title("Forestry action selected", fontdict={'fontsize': 14})
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")

                        cb = fig.colorbar(im, ax=ax, boundaries=boundaries, ticks=range(1, ncat + 1))
                        tick_labels = [treatment_map.get(i, f"ID_{i}") for i in range(1, ncat + 1)]
                        cb.ax.set_yticklabels(tick_labels)

                        plt.tight_layout()


                        # ===== Stacked Bar Plot =====
                        grouped = data_2d.groupby('level_1')['Value'].sum()
                        percentages = grouped / grouped.sum() * 100

                        fig, ax = plt.subplots(figsize=(10, 2))
                        start = 0
                        for level in unique_types:
                            percent = percentages.get(level, 0)
                            ax.barh(0, percent, left=start, color=color_map.get(level,'gray'), label=level)
                            start += percent

                        ax.set_xlim(0, 100)
                        ax.set_yticks([])
                        ax.set_xlabel("Percentage")
                        ax.set_title("Management Decisions (Stacked Proportions)")
                        ax.legend(title="Management Type", bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        plt.show()


                    
                    display(create_map())
                    print("DONE")
                else:
                    display(self.results_output)  # Display new results

        # Link the button to the callback
        optimize_button.on_click(on_optimize_clicked)

        # Initial display of imEps widgets
        #with imEps_container:
        #    display(self.imEps)
        #display(self.fig)
        left_layout = widgets.Layout(width='20%')  # Set the width for the left side (text)
        right_layout = widgets.Layout(width='80%')  # Set the width for the right side (widgets)
        explanation_h= widgets.HTML(
            value="<h2>Instructions</h2>")
        explanation = widgets.HTML(value="<p>This interface allows you to adjust sliders for different variables. You can change the values using the sliders to see how the output changes dynamically. Experiment with the values to see their effects!</p>"
            )
        header = widgets.HTML(value="<h2>Reference point</h2>")
        display(explanation_h) 
        display(explanation)
        display(header) 
        ui = widgets.VBox([widgets.HBox([widgets.Box([explanation_h], layout=left_layout),widgets.Box([header], layout=right_layout)]),widgets.HBox([widgets.Box([explanation], layout=left_layout),widgets.Box([imRef_container], layout=right_layout)])])
        display(imRef_container)
        
        display(HTML("<h2>Constraint values</h2>"))
        imEps_container = widgets.VBox([child for i, child in enumerate(self.imEps.children)])
        ui_const = widgets.VBox([widgets.HBox([widgets.Box([explanation_h], layout=left_layout),widgets.Box([header], layout=right_layout)]),widgets.HBox([widgets.Box([explanation], layout=left_layout),widgets.Box([imEps_container], layout=right_layout)])])
        ui_const = imEps_container
        display(toggle_imEps)
        #imEps_container.clear_output()
        display(widgets.Box([output_container], layout=output_layout))
        
        display(HTML("<h2>Enabled constraints</h2>"))
        display(self.imConst)
        display(optimize_button)
        display(results_placeholder)
        
        remSpatial = widgets.interactive(self.removeSpatialConstraints,{"manual":True,"manual_name": "Remove spatial restriction"})
        #display(remSpatial)

        
    def calculateMFEyvindsonObjectiveRanges(self, debug=False):
        self.debug=True
        lb = {objName:np.inf for objName  in self.eyvindsonObjectives.keys()}
        ub = {objName:-np.inf for objName  in self.eyvindsonObjectives.keys()}
        display_local("Calculating objective ranges")
        for i,objName in enumerate(tqdm(self.eyvindsonObjectives.keys(),file=sys.stdout)):
            self.objectiveFunction = self.solver.Objective()
            print(objName)
            display_local("Optimizing for "+self.eyvindsonObjectives[objName])
            self.objectiveFunction.SetMaximization()
            #if self.eyvindsonObjectives[objName][2] == "max":
            multiplier = 1
            #elif self.eyvindsonObjectives[objName][2] == "min":
            #multiplier = -1
            for objName2 in self.eyvindsonObjectives.keys():
                if objName == objName2:
                    self.objectiveFunction.SetCoefficient(self.eyvindsonObjectives[objName2],multiplier)
                else:
                    # If ranges already calculated also add the other objectives with small coefficients to improve ranges:
                    try:
                        self.objectiveFunction.SetCoefficient(self.eyvindsonObjectives[objName2],multiplier*1e-6/(self.objectiveRangesEY[objName2][1]-self.objectiveRangesEY[objName2][0]))
                    except AttributeError:
                        self.objectiveFunction.SetCoefficient(self.eyvindsonObjectives[objName2],0)                        
            #If we have already been running the GUI, then we need to remove maxDummy from objective function
            try:
                self.objectiveFunction.SetCoefficient(self.maxDummy,0)
            except AttributeError:
                pass
            if self.debug:
                problem = self.solver.ExportModelAsLpFormat(obfuscated=False)
                print(problem,file=open("problem.lp","w"))
            now = datetime.now()
            res = self.solver.Solve()
            time = datetime.now() - now 
            self.solutionTimeStamp = str(now).replace(":"," ")
            if res == self.solver.OPTIMAL:
                display_local("Found an optimal solution in "+str(time.seconds)+" seconds")
                display_local("Objective values are:")
                for i,objName in enumerate(self.eyvindsonObjectives.keys()):
                    display(self.eyvindsonObjectives[objName],self.eyvindsonObjectives[objName].solution_value())
                    if self.eyvindsonObjectives[objName].solution_value() > ub[objName]:
                        ub[objName] = self.eyvindsonObjectives[objName].solution_value()
                    if self.eyvindsonObjectives[objName].solution_value() < lb[objName]:
                        lb[objName] = self.eyvindsonObjectives[objName].solution_value()
            else:
                display("Ratkaisua ei löytynyt")
                if res == self.solver.FIXED_VALUE:
                    display("Objective value fixed")
                if res == self.solver.INFEASIBLE:
                    display("Ongelma ei ole ratkaistavissa")
                if res == self.solver.ABNORMAL:
                    display("Something strange in the problem")
                if res == self.solver.NOT_SOLVED:
                    display("Problem could not be solved for some reason")            
        self.objectiveRangesEY = {objName: (lb[objName],ub[objName]) for objName in self.eyvindsonObjectives.keys()}           

    def solveMultifunctionality(self):
        self.objectiveFunction = self.solver.Objective()
        self.objectiveFunction.SetMaximization()
        multiplier = 1
        for objName2 in self.eyvindsonObjectives.keys():
            self.objectiveFunction.SetCoefficient(self.eyvindsonObjectives[objName2],multiplier/(self.objectiveRangesEY[objName2][1]-self.objectiveRangesEY[objName2][0])) #MAYBE need to objective - minimum to properly normalize in objective function
        now = datetime.now()
        res = self.solver.Solve()
        time = datetime.now() - now
        
        if res == self.solver.OPTIMAL:
            display_local("Found an optimal solution in "+str(time.seconds)+" seconds")
            display_local("Objective values are:")
            for i,objName in enumerate(self.eyvindsonObjectives.keys()):
                display(self.eyvindsonObjectives[objName],self.eyvindsonObjectives[objName].solution_value())
        else:
            display("Ratkaisua ei löytynyt")
            if res == self.solver.FIXED_VALUE:
                display("Objective value fixed")
            if res == self.solver.INFEASIBLE:
                display("Ongelma ei ole ratkaistavissa")
            if res == self.solver.ABNORMAL:
                display("Something strange in the problem")
            if res == self.solver.NOT_SOLVED:
                display("Problem could not be solved for some reason")  

    def addEyvindsonMultifunctionality(self, ecoSystemServices, aggregation = None):
        #IF no aggregation method is specified -- then all will be assumed to be the average method
        if aggregation is None:
            aggregation = {}
            for key in ecoSystemServices.keys():
                aggregation[key] = "AVG"

        #Check if dictionary exists, it not create
        self.eyvindsonObjectives = dict()
        
        #CONSTRUCT value to indicate RANGE of MF -- one per ESS / BD
        for key in ecoSystemServices.keys():
            self.eyvindsonObjectives[key] = self.solver.NumVar(-self.solver.infinity(),self.solver.infinity(),"MF"+str(key))
            if aggregation[key]=="AVG":
                self.solver.Add(self.eyvindsonObjectives[key] == sum((self.objective[objName]-self.objectiveRanges[objName][0])/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]) for objName in ecoSystemServices[key].keys())/len(ecoSystemServices[key].keys()),name="EyvindsonObjective"+key)
            elif aggregation[key]=="MIN":
                for objName in ecoSystemServices[key].keys():
                    self.solver.Add(self.eyvindsonObjectives[key]<=(self.objective[objName]-self.objectiveRanges[objName][0])/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0]),name="EyvindsonObjective"+key+objName) # CHECK __ IS THIS MAXIMIZING THE MINIMUM OR MINIMIZING THE MAXIMUM?
            #self.solver.Add(self.objective[objName]<=self.objectivesByYear[(objName,year)])
            #for objName in self.ecoSystemServices[key].keys():
            #    self.solver.Add(self.eyvindsonObjectives[key] >= sum((self.objective[objName]-self.objectiveRanges[objName][0])/(self.objectiveRanges[objName][1]-self.objectiveRanges[objName][0])),name="EyvindsonObjective"+key+objName)
        #CALCLATE RANGES for ESS Groups:
        self.calculateMFEyvindsonObjectiveRanges()
