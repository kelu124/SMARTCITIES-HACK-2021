## Overview

* TO USE
  * OSM all roads (SingaporeStreets.zip)
  * TREES
  * CCTV
  * CAMERAS
* Streamlit, Folium 
* Networkx for solution finding
* Oracle cloud for VM

* Perimeter: center of singapore

### Route planners

#### Libs & examples

* https://ipython-books.github.io/147-creating-a-route-planner-for-a-road-network/ 
* https://www.geeksforgeeks.org/python-program-for-dijkstras-shortest-path-algorithm-greedy-algo-7/
* https://www.codementor.io/blog/basic-pathfinding-explained-with-python-5pil8767c1
* https://shakasom.medium.com/routing-street-networks-find-your-way-with-python-9ba498147342
* https://pypi.org/project/pathfinding/
* https://github.com/OanaGaskey/Route-Planner

#### Experiments 

* https://github.com/mottmacdonaldglobal/SMARTCITIES-HACK-2021/tree/routeplanner-experiments

### Recommender 

* https://towardsdatascience.com/pmf-for-recommender-systems-cbaf20f102f0

### Walkability

* https://medium.com/data-mining-the-city/simulation-visualization-of-walkability-score-ef88d5893301


### Apis at large

* https://github.com/jlowe000/api-pipeline
* https://www.walkscore.com/professional/api-sample-code.php

## Solutions

### Viz

* plotly 
* https://kepler.gl/ -- eg demo at https://www.kaggle.com/mariamingallon/kepler-gl-hex-data-demo
* streamlit https://docs.streamlit.io/en/stable/deploy_streamlit_app.html
  * demo by Maria: https://share.streamlit.io/mariamingallonmm/smartcities-hack-streamlitapp


### Data:

* postgis+postgresql 
* geopandas
* networkx


## Example of code:

* School picker: https://github.com/datagovsg/school-picker
* Walk for food : https://www.dshkol.com/post/will-walk-for-food-exploring-singapore-hawker-centre-density/

## Oracle

* https://apex.oracle.com/en/platform/low-code/
* https://pretius.com/integration-of-google-assistant-with-oracle-apex-part-1/


## SG / Data sources

### Github for SG DATA

* https://github.com/datagovsg
* https://github.com/datagovsg/ckan
* https://www.openstreetmap.org/ for data

### Route data

* https://www.sla.gov.sg/geospatial-development-and-services/onemap
* https://datamall.lta.gov.sg/content/datamall/en.html
* https://storymaps.arcgis.com/stories/4f782ddd18914d16bc912a81ef83ac80
* https://data.gov.sg/dataset/sdcp-park-connector-line
* https://www.onemap.gov.sg/home/
* https://data.gov.sg/dataset/master-plan-2019-road-name-layer?resource_id=667b3f04-38d7-4be1-bb0e-6aa4141a03ee

### Secondary data

#### Weather

* Sun orientation - UV index https://data.gov.sg/dataset/ultraviolet-index-uvi 
* Weather forecast (real time data) - https://data.gov.sg/dataset/realtime-weather-readings 
* Humidity & temperature -  https://data.gov.sg/dataset/realtime-weather-readings 
* Air quality API : https://data.gov.sg/dataset/pm2-5 

#### Walkability

* https://sg.mapometer.com/walking

#### Green

* Nature parks (data.gov.sg) - https://data.gov.sg/dataset/parkssg 
* Park connections (data.gov.sg)
* Park Connector Loop - https://cloud.csiss.gmu.edu/uddi/nl/dataset/park-connector-loop 
* Historic green roads https://cloud.csiss.gmu.edu/uddi/dataset/heritage-road-green-buffers 
* Vertical greenery - https://cloud.csiss.gmu.edu/uddi/dataset/nparks-skyrise-greenery 
* Trees.sg
* Used by https://github.com/cheeaun/exploretrees-sg (Nice viz)

#### Available curated content

* https://www.justrunlah.com/singapore-running-routes/ 
* http://www.yoursingapore.com/ 


### Walking

* https://sg.mapometer.com/walking
