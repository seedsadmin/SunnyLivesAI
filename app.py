from fileinput import filename 
from flask import *  
from pathlib import Path
import os
from create_tiles import process_file
from generate_footprints import gen_foot
from simplify import simplify_footprints
from classify_footprints import classify_roof
from risk_calc import calculate_risk
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

import pathlib


@app.route('/pdf/<path:filename>/', methods=['GET', 'POST'])
def getrisk(filename):
    project_id = request.args.get('project_id', "1")
    project_name = request.args.get('project_name', "1")
    dirdata = 'uploads/'+ project_id +'/riskscore/output'
    file_name = project_name +'_final_house_level_4326.geojson'
    return send_from_directory(directory=dirdata, path=file_name)

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files = request.files.getlist("file[]")
    
    project_id = request.form['pid']
    projection = request.form['projection']

    str_path = 'uploads/' + project_id
    dir_path = Path(str_path)

    # Check if directory exists, if not, create it
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
   
    for file in uploaded_files:
        if file:
            # Process each file

            file.save(str_path  + '/' + file.filename)
    return 'Files successfully uploaded'


@app.route('/process', methods=['POST'])
def process():
   
    project_id = request.json['pid']
    projection = request.json['projection']

    process_file(project_id)
   
    return 'Files successfully uploaded'


@app.route('/genfoot', methods=['POST'])
def genfoot():
   
    project_id = request.json['pid']
    projection = request.json['projection']
    region_name = request.json['region_name']
    gen_foot(project_id, region_name)
   
    return 'Files successfully uploaded'


@app.route('/simplify', methods=['POST'])
def simpl():
   
    project_id = request.json['pid']
    projection = request.json['projection']
    region_name = request.json['region_name']
    simplify_footprints(project_id, region_name, projection)
   
    return 'Files successfully uploaded'

@app.route('/classify', methods=['POST'])
def classif():
   
    project_id = request.json['pid']
    projection = request.json['projection']
    region_name = request.json['region_name']
    analytic_utm = request.json['analytic_utm']

    classify_roof(project_id, region_name, analytic_utm, projection)
   
    return 'Files successfully uploaded'

@app.route('/risk_calc', methods=['POST'])
def risk():
   
    project_id = request.json['pid']
    projection = request.json['projection']
    region_name = request.json['region_name']
    analytic_utm = request.json['analytic_utm']

    LCString=request.json['LCString']
    ri_water_poly=request.json['ri_water_poly']
    ri_water_line=request.json['ri_water_line']
    ri_ocean=request.json['ri_ocean']
    ri_twi=request.json['ri_twi']
    ri_elevation=request.json['ri_elevation']
    ri_ndvi=request.json['ri_ndvi']
    ri_landslide_risk=request.json['ri_landslide_risk']
    ri_impervious=request.json['ri_impervious']
    ri_footprint_label=request.json['ri_footprint_label']
    ri_footprint_area=request.json['ri_footprint_area']



    calculate_risk(project_id, region_name, analytic_utm, projection,
                   LCString, ri_water_poly,
                   ri_water_line, ri_ocean, ri_twi, ri_elevation, ri_ndvi, ri_landslide_risk, 
                   ri_impervious, ri_footprint_label, ri_footprint_area)
   
    return 'Files successfully uploaded'

@app.route('/all_data', methods=['POST'])
def alldata():
   
    project_id = request.json['pid']
    projection = request.json['projection']
    region_name = request.json['region_name']
    analytic_utm = request.json['analytic_utm']

    
    LCString=request.json['LCString']
    ri_water_poly=request.json['ri_water_poly']
    ri_water_line=request.json['ri_water_line']
    ri_ocean=request.json['ri_ocean']
    ri_twi=request.json['ri_twi']
    ri_elevation=request.json['ri_elevation']
    ri_ndvi=request.json['ri_ndvi']
    ri_landslide_risk=request.json['ri_landslide_risk']
    ri_impervious=request.json['ri_impervious']
    ri_footprint_label=request.json['ri_footprint_label']
    ri_footprint_area=request.json['ri_footprint_area']



    process_file(project_id)
    gen_foot(project_id, region_name)
    simplify_footprints(project_id, region_name, projection)
    
    classify_roof(project_id, region_name, analytic_utm, projection)
   

   

    calculate_risk(project_id, region_name, analytic_utm, projection,
                   LCString, ri_water_poly,
                   ri_water_line, ri_ocean, ri_twi, ri_elevation, ri_ndvi, ri_landslide_risk, 
                   ri_impervious, ri_footprint_label, ri_footprint_area)
    
    return 'Files successfully uploaded'


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=80)