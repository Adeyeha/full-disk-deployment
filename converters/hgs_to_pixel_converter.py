# Global Definitions
import math

CDELT = 0.600000023842;
HPCCENTER = 4096.0 / 2.0;

rsun_meters = 696000000.0;
dsun_meters = 149597870691.0;


def convert_hgs_to_pix(x, y):
	hcc_x, hcc_y = convertHG_HCC(x, y)
	hpc_x, hpc_y = convertHCC_HPC(hcc_x, hcc_y)
	
	pix_x = HPCCENTER + (hpc_x/CDELT)
	pix_y = HPCCENTER - (hpc_y/CDELT)
	
	return int(pix_x/8.0), int(pix_y/8.0)

def convertHG_HCC(hglon_deg, hglat_deg):
	b0_deg = 0
	l0_deg = 0
	lon = math.radians(hglon_deg)
	lat = math.radians(hglat_deg)
	
	cosb = math.cos(math.radians(b0_deg))
	sinb = math.sin(math.radians(b0_deg))
	
	lon = lon - math.radians(l0_deg)
	
	cosx = math.cos(lon)
	sinx = math.sin(lon)
	cosy = math.cos(lat)
	siny = math.sin(lat)

	#Perform the conversion.
	x = rsun_meters * cosy * sinx
	y = rsun_meters * (siny * cosb - cosy * cosx * sinb)
	
	return x, y
	

def convertHCC_HPC(x, y):
# 	print("HCC: ",  x, y)
# 	print(math.pow(rsun_meters, 2) - math.pow(x, 2) - math.pow(y, 2))
	try:
		z = math.sqrt(math.pow(rsun_meters, 2) - math.pow(x, 2) - math.pow(y, 2))
	except ValueError:
		z=0

	zeta = dsun_meters - z
	distance = math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(zeta, 2))
	hpcx = math.degrees(math.atan2(x, zeta))
	hpcy = math.degrees(math.asin(y / distance))

	#if angle_units == 'arcsec' :
	hpcx = 60 * 60 * hpcx
	hpcy = 60 * 60 * hpcy

	return hpcx, hpcy


# ### Reverse Conversion
# def convert_pix_to_hgs(pix_x, pix_y):
#     hpc_x, hpc_y = convert_pix_to_HPC(pix_x, pix_y)
#     hcc_x, hcc_y = convertHPC_to_HCC(hpc_x, hpc_y)
#     hglon_deg, hglat_deg = convertHCC_to_HG(hcc_x, hcc_y)

#     return hglon_deg, hglat_deg


# def convert_pix_to_HPC(pix_x, pix_y):
#     hpc_x = (pix_x * 8.0 - HPCCENTER) * CDELT
#     hpc_y = (HPCCENTER - pix_y * 8.0) * CDELT

#     return hpc_x, hpc_y


# # def convertHPC_to_HCC(hpc_x, hpc_y):
# #     hpc_x_rad = hpc_x / (60 * 60)
# #     hpc_y_rad = hpc_y / (60 * 60)

# #     distance = dsun_meters
# #     zeta = distance * math.cos(math.radians(hpc_y_rad))

# #     x = distance * math.sin(math.radians(hpc_y_rad)) * math.sin(math.radians(hpc_x_rad))
# #     y = distance * math.sin(math.radians(hpc_y_rad)) * math.cos(math.radians(hpc_x_rad))

# #     return x, y


# def convertHPC_to_HCC(hpc_x, hpc_y):
#     hpc_x_rad = math.radians(hpc_x / (60 * 60))
#     hpc_y_rad = math.radians(hpc_y / (60 * 60))

#     r = dsun_meters * math.cos(hpc_y_rad)
#     z = dsun_meters * math.sin(hpc_y_rad)

#     x = r * math.sin(hpc_x_rad)
#     y = r * math.cos(hpc_x_rad)

#     return x, y



# # def convertHCC_to_HG(hcc_x, hcc_y):
# #     b0_deg = 0
# #     l0_deg = 0

# #     cosb = math.cos(math.radians(b0_deg))
# #     sinb = math.sin(math.radians(b0_deg))

# #     r = math.sqrt(hcc_x**2 + hcc_y**2)
# #     z = math.sqrt(rsun_meters**2 - r**2)

# #     distance = math.sqrt(r**2 + (dsun_meters - z)**2)
# #     lat = math.asin(hcc_y / distance)
# #     lon = math.atan2(hcc_x, z) + math.radians(l0_deg)

# #     hglon_deg = math.degrees(lon)
# #     hglat_deg = math.degrees(lat)

# #     return hglon_deg, hglat_deg


# def convertHCC_to_HG(hcc_x, hcc_y):
#     l0_deg = 0  # Define the observer's longitude here
    
#     r = math.sqrt(hcc_x ** 2 + hcc_y ** 2)

#     try:
#         z = math.sqrt(rsun_meters**2 - r**2)
#     except ValueError:
#         z = 0

#     zeta = dsun_meters - z
#     hpc_x = math.atan2(hcc_x, zeta)
#     hpc_y = math.asin(hcc_y / math.sqrt(hcc_x ** 2 + hcc_y ** 2 + zeta ** 2))

#     lon = hpc_x + math.radians(l0_deg)
#     lat = hpc_y

#     hglon_deg = math.degrees(lon)
#     hglat_deg = math.degrees(lat)

#     return hglon_deg, hglat_deg


