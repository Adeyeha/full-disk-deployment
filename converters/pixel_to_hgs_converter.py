import math

CDELT = 0.6  # arcseconds per pixel
HPCCENTER = 4096 / 2
dsun_meters = 1.496e11
rsun_meters = 6.9634e8

def convert_pix_to_hgs(pix_x, pix_y):
    hpc_x, hpc_y = convert_pix_to_HPC(pix_x, pix_y)
    hcc_x, hcc_y = convertHPC_to_HCC(hpc_x, hpc_y)
    hglon_deg, hglat_deg = convertHCC_to_HG(hcc_x, hcc_y)

    return hglon_deg, hglat_deg

def convert_pix_to_HPC(pix_x, pix_y):
    hpc_x = (pix_x - HPCCENTER) * CDELT
    hpc_y = (HPCCENTER - pix_y) * CDELT

    return hpc_x, hpc_y

def convertHPC_to_HCC(hpc_x, hpc_y):
    hpc_x_rad = math.radians(hpc_x / (60 * 60))
    hpc_y_rad = math.radians(hpc_y / (60 * 60))

    r = dsun_meters * math.cos(hpc_y_rad)
    z = dsun_meters * math.sin(hpc_y_rad)

    x = r * math.sin(hpc_x_rad)
    y = r * math.cos(hpc_x_rad)

    return x, y

def convertHCC_to_HG(hcc_x, hcc_y):
    l0_deg = 0  # Define the observer's longitude here

    r = math.sqrt(hcc_x ** 2 + hcc_y ** 2)

    try:
        z = math.sqrt(rsun_meters**2 - r**2)
    except ValueError:
        z = 0

    zeta = dsun_meters - z
    hpc_x = math.atan2(hcc_x, zeta)
    hpc_y = math.asin(hcc_y / math.sqrt(hcc_x ** 2 + hcc_y ** 2 + zeta ** 2))

    lon = hpc_x + math.radians(l0_deg)
    lat = hpc_y

    hglon_deg = math.degrees(lon)
    hglat_deg = math.degrees(lat)

    return hglon_deg, hglat_deg
