from osgeo import gdal
import ee
import time
import numpy as np

# --- Authenticate and Initialize GEE ---
try:
    ee.Initialize(project='coe-aiml-b8')
    print("✅ Earth Engine Initialized Successfully!")
except Exception as e:
    print(f"❌ Earth Engine init failed: {e}")
    exit(1)

# --- USER-DEFINED PARAMETERS ---
# Yamuna River, Noida stretch
AOI = ee.Geometry.Rectangle([77.30, 28.50, 77.40, 28.60])

START_DATE = '2023-01-01'
END_DATE = '2024-12-31'
BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Blue, Green, Red, NIR, SWIR

EXPORT_FOLDER = 'ISMD_SATELLITE_DATA'
SCALE = 10
CRS = 'EPSG:4326'

# --- CLOUD MASKING FUNCTION ---
def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 L2A image using the SCL band."""
    qa = image.select('SCL')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000).select(BANDS).copyProperties(image, ["system:time_start"])

# --- SPECTRAL INDICES FUNCTIONS ---
def add_spectral_indices(image):
    """Calculates and adds NDVI, NDWI, and MNDWI bands to an image."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')  # Green - NIR
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')  # Green - SWIR1
    return image.addBands([ndvi, ndwi, mndwi])

# --- MAIN PROCESSING LOGIC ---
def get_monthly_composites(start_date, end_date, aoi):
    """Generates a collection of cloud-free monthly composites."""
    image_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .map(mask_s2_clouds)

    monthly_composites = []
    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_start = ee.Date.fromYMD(year, month, 1)
            month_end = month_start.advance(1, 'month')
            
            monthly_collection = image_collection.filterDate(month_start, month_end)
            
            if monthly_collection.size().getInfo() > 0:
                median_composite = monthly_collection.median().clip(aoi)
                composite_with_indices = add_spectral_indices(median_composite)
                composite_with_indices = composite_with_indices.set({
                    'year': year,
                    'month': month,
                    'system:time_start': month_start.millis()
                })
                monthly_composites.append(composite_with_indices)

    return ee.ImageCollection.fromImages(monthly_composites)

# --- EXECUTION ---
print("🚀 Starting satellite data collection...")
print(f"Area: Yamuna River, Noida ({AOI.getInfo()['coordinates']})")
print(f"Date Range: {START_DATE} to {END_DATE}")

monthly_collection = get_monthly_composites(START_DATE, END_DATE, AOI)
composite_count = monthly_collection.size().getInfo()
print(f"📊 Generated {composite_count} monthly composites")

# --- EXPORT TO GOOGLE DRIVE ---
print("📤 Starting export to Google Drive...")
image_list = monthly_collection.toList(monthly_collection.size())

for i in range(composite_count):
    image = ee.Image(image_list.get(i))
    year = image.get('year').getInfo()
    month = image.get('month').getInfo()
    
    filename = f'S2_Yamuna_{year}_{month:02d}'
    print(f"  Exporting {filename}...")
    
    task = ee.batch.Export.image.toDrive(
        image=image.select(['B4', 'B3', 'B2', 'NDVI', 'NDWI', 'MNDWI']),  # RGB + indices
        description=filename,
        folder=EXPORT_FOLDER,
        fileNamePrefix=filename,
        region=AOI,
        scale=SCALE,
        crs=CRS
    )
    task.start()
    time.sleep(2)  # Avoid overwhelming GEE servers

print(f"✅ All {composite_count} export tasks submitted!")
print("📁 Check your Google Drive folder: 'ISMD_SATELLITE_DATA'")
print("⏳ Exports will take some time. Check GEE Task Manager for progress.")