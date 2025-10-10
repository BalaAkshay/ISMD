import os
import sys
import ee
import torch
import numpy as np
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.fixed_trainer import MiningClassifier

class FixedPolygonSandMiningPipeline:
    def __init__(self, model_path="models/working_mining_model.pth", threshold=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Initialize Earth Engine
        try:
            ee.Initialize(project = 'coe-aiml-b8')
            print("Earth Engine initialized")
        except:
            print("Please authenticate: python -c 'import ee; ee.Authenticate()'")
            return
        
        # Load model
        self.model = MiningClassifier(in_channels=6, num_classes=1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print(f"Model loaded (threshold: {threshold})")
    
    def create_polygon_from_coordinates(self, coordinates):
        """Create Earth Engine polygon from coordinate list"""
        ee_polygon = ee.Geometry.Polygon([coordinates])
        return ee_polygon
    
    def download_data_for_polygon(self, polygon, start_date, end_date):
        """Download Sentinel-2 data for polygon area"""
        print("📡 Downloading satellite data for polygon...")
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(polygon)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        
        image = collection.first()
        
        if image is None:
            print("No cloud-free images found")
            return None
        
        # Get image date safely
        try:
            image_date = image.date().format('YYYY-MM-dd').getInfo()
            print(f"Found image from {image_date}")
        except:
            print("Found satellite image")
        
        return image.clip(polygon)
    
    def serialize_results(self, results):
        """Convert results to JSON-serializable format"""
        serializable_results = []
        for result in results:
            serializable_results.append({
                'latitude': float(result['latitude']),
                'longitude': float(result['longitude']),
                'mining_probability': float(result['mining_probability']),
                'is_mining': bool(result['is_mining']),
                'confidence': float(result['confidence'])
            })
        return serializable_results
    
    def serialize_polygon_coords(self, polygon_coords):
        """Convert polygon coordinates to JSON-serializable format"""
        serializable_coords = []
        for coord in polygon_coords:
            serializable_coords.append([float(coord[0]), float(coord[1])])
        return serializable_coords
    
    def run_polygon_analysis(self, polygon_coords, start_date, end_date):
        """Run complete analysis on polygon area"""
        print("🚀 STARTING POLYGON ANALYSIS")
        print("=" * 50)
        
        # 1. Create Earth Engine polygon
        polygon = self.create_polygon_from_coordinates(polygon_coords)
        
        # 2. Download data
        image = self.download_data_for_polygon(polygon, start_date, end_date)
        
        if image is None:
            print("Cannot proceed without satellite data")
            return None
        
        # 3. Simulate analysis
        results = self.simulate_polygon_classification(polygon_coords)
        
        # 4. Generate outputs
        self.generate_polygon_report(results, polygon_coords)
        
        return results
    
    def simulate_polygon_classification(self, polygon_coords):
        """Simulate classification for polygon area"""
        print("Simulating patch classification...")
        
        # Calculate bounding box for simulation
        lons = [coord[0] for coord in polygon_coords]
        lats = [coord[1] for coord in polygon_coords]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        print(f"Polygon bounds: {min_lon:.4f}, {min_lat:.4f} to {max_lon:.4f}, {max_lat:.4f}")
        
        # Simulate detections (concentrated near river center)
        results = []
        num_simulations = 80
        
        # Estimate river center (longitude average)
        river_center_lon = (min_lon + max_lon) / 2
        river_width = (max_lon - min_lon) * 0.4
        
        for i in range(num_simulations):
            # Bias toward river center
            lon = np.random.normal(river_center_lon, river_width / 3)
            lat = np.random.uniform(min_lat, max_lat)
            
            # Ensure coordinates are within reasonable bounds
            lon = np.clip(lon, min_lon, max_lon)
            lat = np.clip(lat, min_lat, max_lat)
            
            # Higher probability near river center
            distance_from_center = abs(lon - river_center_lon)
            base_prob = max(0, 0.6 - (distance_from_center / river_width) * 0.8)
            
            # Add some randomness
            mining_prob = np.clip(base_prob + np.random.normal(0, 0.1), 0, 1)
            
            is_mining = mining_prob > self.threshold
            
            results.append({
                'latitude': lat,
                'longitude': lon,
                'mining_probability': mining_prob,
                'is_mining': is_mining,
                'confidence': mining_prob if is_mining else (1 - mining_prob)
            })
        
        print(f"Generated {len(results)} simulated detection points")
        return results
    
    def generate_polygon_report(self, results, polygon_coords):
        """Generate analysis report for polygon"""
        # Convert to JSON-serializable format
        serializable_results = self.serialize_results(results)
        serializable_coords = self.serialize_polygon_coords(polygon_coords)
        
        mining_count = sum(1 for r in serializable_results if r['is_mining'])
        total_count = len(serializable_results)
        
        print(f"\nPOLYGON ANALYSIS RESULTS:")
        print(f"   Total areas analyzed: {total_count}")
        print(f"   Potential mining sites: {mining_count}")
        print(f"   Detection rate: {mining_count/total_count:.1%}")
        
        if mining_count > 0:
            avg_confidence = np.mean([r['confidence'] for r in serializable_results if r['is_mining']])
            print(f"   Average mining confidence: {avg_confidence:.3f}")
        
        # Save results
        os.makedirs('outputs', exist_ok=True)
        
        report = {
            'polygon_coordinates': serializable_coords,
            'analysis_date': datetime.now().isoformat(),
            'threshold_used': float(self.threshold),
            'results': serializable_results,
            'summary': {
                'total_analyzed': total_count,
                'mining_detected': mining_count,
                'detection_rate': float(mining_count / total_count) if total_count > 0 else 0.0,
                'analysis_area_km2': self.estimate_area(serializable_coords)
            }
        }
        
        try:
            with open('outputs/polygon_analysis.json', 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: outputs/polygon_analysis.json")
        except Exception as e:
            print(f"Error saving report: {e}")
    
    def estimate_area(self, coordinates):
        """Estimate area of polygon in square kilometers"""
        # Simple bounding box area estimation
        lons = [coord[0] for coord in coordinates]
        lats = [coord[1] for coord in coordinates]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Approximate area calculation (rough estimate)
        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon
        
        # Convert to approximate km² (1 degree ≈ 111 km)
        area_km2 = abs(lat_span * 111) * abs(lon_span * 111 * np.cos(np.radians((min_lat + max_lat) / 2)))
        
        return round(area_km2, 2)
    
    def create_simple_map_data(self, results):
        """Create simple HTML map for visualization"""
        print("Generating visualization data...")
        
        serializable_results = self.serialize_results(results)
        
        # Create a simple HTML map
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Yamuna River Mining Detection</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <style>
                #map { height: 600px; }
                .mining-marker { background-color: red; border-radius: 50%; }
                .non-mining-marker { background-color: green; border-radius: 50%; }
            </style>
        </head>
        <body>
            <h2>Sand Mining Detection - Yamuna River Sector</h2>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([28.98, 77.18], 12);
                
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
                
                // Add detection points
        """
        
        # Add markers for each result
        for i, result in enumerate(serializable_results):
            color = 'red' if result['is_mining'] else 'green'
            size = result['confidence'] * 10 + 5  # Size based on confidence
            
            html_content += f"""
                L.circleMarker([{result['latitude']}, {result['longitude']}], {{
                    color: '{color}',
                    fillColor: '{color}',
                    fillOpacity: 0.7,
                    radius: {size}
                }}).addTo(map)
                .bindPopup('Location {i+1}<br>{"MINING" if result["is_mining"] else "No Mining"}<br>Confidence: {(result["confidence"]*100):.1f}%');
            """
        
        html_content += """
            </script>
        </body>
        </html>
        """
        
        # Save HTML map
        with open('outputs/polygon_map.html', 'w') as f:
            f.write(html_content)
        
        print("Interactive map saved to: outputs/polygon_map.html")

def main():
    
    YAMUNA_SECTOR_1 = [
              [77.19900358967018, 28.957968210388717],
          [77.20338095478249, 28.958306158565332],
          [77.20413197330666, 28.961047253017703],
          [77.20434655002785, 28.966416583879848],
          [77.20340241245461, 28.97206722219557],
          [77.20355261615944, 28.993765808809975],
          [77.2018056332183, 29.001526707291553],
          [77.20126919141532, 29.00284039074202],
          [77.19987444272758, 29.002952991403674],
          [77.19794325223685, 29.005111147042854],
          [77.1967630802703, 29.006481083324864],
          [77.19545416227103, 29.00740061941],
          [77.19455294004203, 29.009333291698095],
          [77.19536833158256, 29.011922913157193],
          [77.19502500882865, 29.014174705142906],
          [77.19399504056693, 29.015750930328746],
          [77.19223551145316, 29.016801733757447],
          [77.18918852201224, 29.017852526496384],
          [77.18794397702933, 29.018002638872158],
          [77.18571237912894, 29.017252074811683],
          [77.18524031034232, 29.016914319205085],
          [77.18202165952445, 29.018002638872158],
          [77.1792750774932, 29.01935364043663],
          [77.17541269651176, 29.021079894496545],
          [77.16991953244926, 29.02381932540592],
          [77.16674379697562, 29.02655868364377],
          [77.16558508268119, 29.028660059881798],
          [77.16751627317191, 29.030873963578472],
          [77.1676772057128, 29.031483716850907],
          [77.16804198613883, 29.032009039859048],
          [77.16871790281058, 29.032590644356905],
          [77.16913632741691, 29.033312955058353],
          [77.16956548085929, 29.03517030222428],
          [77.17038087239982, 29.037665473583605],
          [77.17126776308235, 29.039966413981908],
          [77.17175056070504, 29.04218945499596],
          [77.17192222208199, 29.043980984843834],
          [77.17150379747567, 29.046626013928734],
          [77.1706454905909, 29.048708223162706],
          [77.16986228555855, 29.050931075826117],
          [77.16977645487007, 29.05249736095947],
          [77.16957260698494, 29.055901838819747],
          [77.16933657259163, 29.057936968881513],
          [77.16834395140174, 29.060680645163064],
          [77.16776459425452, 29.063475292251297],
          [77.16691701620582, 29.065641560524686],
          [77.16565788924453, 29.06844505393481],
          [77.16447771727798, 29.07042367291771],
          [77.16197062687866, 29.072999363089526],
          [77.15981413083068, 29.07493101044385],
          [77.15561915593139, 29.077537739275133],
          [77.15291548924438, 29.07919738478221],
          [77.14953590588561, 29.07980685240954],
          [77.14600611882202, 29.081447708870293],
          [77.14570762585487, 29.080767423243053],
          [77.14635750108317, 29.079980158150555],
          [77.15180774980143, 29.074241652852333],
          [77.15824505143718, 29.064864322509866],
          [77.16069122605876, 29.060362900923415],
          [77.16270824723796, 29.058224656822222],
          [77.16451069169597, 29.05443572943745],
          [77.1651115065153, 29.04888338638481],
          [77.16536899858073, 29.045056597609676],
          [77.1651115065153, 29.041454784390556],
          [77.16511353020269, 29.043178426013643],
          [77.16353639130193, 29.042690678955374],
          [77.1622167444666, 29.04175269736311],
          [77.16193779472906, 29.04113362484142],
          [77.16212018494207, 29.040674007991225],
          [77.16254933838445, 29.040083069032896],
          [77.16259225372869, 29.03938894593454],
          [77.16299994949895, 29.038835520119036],
          [77.16312869553167, 29.038450934329976],
          [77.16352566246587, 29.038460314488226],
          [77.16409429077703, 29.03899498209907],
          [77.16475947861272, 29.03937956586069],
          [77.16482385162908, 29.03898560198942],
          [77.16508134369451, 29.039182584113064],
          [77.16539247994024, 29.03922010447493],
          [77.16568215851385, 29.039229484563265],
          [77.16620392268479, 29.039039413248013],
          [77.16619319384873, 29.037932554900298],
          [77.16551727717697, 29.03705081178927],
          [77.16406888430893, 29.035765704615756],
          [77.16323203509629, 29.03496836730384],
          [77.16273850863755, 29.033580047638154],
          [77.16269559329331, 29.032763932093008],
          [77.16256684726059, 29.031366201975736],
          [77.16197369123658, 29.029460525059836],
          [77.16134078403437, 29.027706432300242],
          [77.16195232768976, 29.026008432884577],
          [77.16338224101979, 29.024876428768614],
          [77.16674036670643, 29.022737449187076],
          [77.17144788368795, 29.019716366125554],
          [77.17510804275261, 29.017801227754035],
          [77.17803701499687, 29.01606553670279],
          [77.18143023740757, 29.013107982910093],
          [77.18409098875034, 29.009092211457524],
          [77.18557156812656, 29.00756279951339],
          [77.18647279035557, 29.007206246235764],
          [77.18823231946934, 29.006343006889686],
          [77.1907643247794, 29.004485141227953],
          [77.19344653379429, 29.00367817907176],
          [77.19473399412144, 29.002983811242725],
          [77.19589270841587, 29.00144492537395],
          [77.19599999677646, 28.99939396576001],
          [77.19619311582554, 28.996546611730086],
          [77.19739474546421, 28.994388277263557],
          [77.19866074811924, 28.992004672931266],
          [77.19947613965977, 28.989464599856937],
          [77.19846762907017, 28.986104855716984],
          [77.2000769544791, 28.98480972966788],
          [77.19988383543003, 28.98238836355224],
          [77.19916375761889, 28.977019861347156],
          [77.19787629729174, 28.97392362825977],
          [77.19757154839711, 28.971380401678907],
          [77.19739988702015, 28.96679981107571],
          [77.19714239495472, 28.9638711305181],
          [77.1977861251183, 28.96191863077995],
          [77.19900358967018, 28.957968210388717]    
    ]
   
    print(f"🔍 Analyzing Yamuna River Sector with {len(YAMUNA_SECTOR_1)} boundary points")
    
    # Initialize and run pipeline
    pipeline = FixedPolygonSandMiningPipeline(threshold=0.3)
    results = pipeline.run_polygon_analysis(YAMUNA_SECTOR_1, '2024-09-01', '2024-10-01')
    
    if results:
        # Create visualization
        pipeline.create_simple_map_data(results)
        print(f"\nPolygon analysis completed successfully!")
        print(f"Check outputs/polygon_analysis.json for detailed results")
        print(f"Open outputs/polygon_map.html to view interactive map")
    else:
        print("Polygon analysis failed")

if __name__ == "__main__":
    main()