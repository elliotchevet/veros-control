"""Proposed rectangular basin domains, for discussion before implementing masks."""
from pathlib import Path
import os
root=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(root/'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy
cartopy.config['data_dir']=root/'cartopy_data'
cartopy.config['pre_existing_data_dir']=root/'cartopy_data'
import cartopy.crs as ccrs
import cartopy.feature as cfeature
fig=plt.figure(figsize=(15,8.5),facecolor='white')
ax=fig.add_axes([.055,.20,.89,.70],projection=ccrs.PlateCarree())
ax.set_global();ax.set_facecolor('white')
ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#bcbcbc',edgecolor='none',zorder=1)
ax.coastlines(resolution='110m',linewidth=.35,color='#777777',zorder=2)
boxes=[(-180,0,110,90),(120,0,60,90),(-180,-60,110,60),(120,-60,60,60),(-70,0,90,90),(-70,-60,90,60),(20,-60,100,90),(-180,-90,360,30)]
for x,y,w,h in boxes:ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='black',linewidth=1.7,transform=ccrs.PlateCarree(),zorder=4))
for label,x,y in [('NORTH\nPACIFIC',-135,35),('NORTH\nPACIFIC',151,35),('SOUTH\nPACIFIC',-135,-28),('SOUTH\nPACIFIC',151,-28),('NORTH\nATLANTIC',-30,32),('SOUTH\nATLANTIC',-28,-28),('INDIAN',70,-25),('SOUTHERN',0,-73)]:
 ax.text(x,y,label,ha='center',va='center',fontsize=12,fontweight='bold',transform=ccrs.PlateCarree(),bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=3),zorder=5)
ax.set_xticks([-180,-120,-70,0,20,120,180],crs=ccrs.PlateCarree())
ax.set_yticks([-90,-60,0,30,90],crs=ccrs.PlateCarree())
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
ax.xaxis.set_major_formatter(LongitudeFormatter());ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(labelsize=10)
fig.suptitle('Proposed rectangular ocean-basin domains',fontsize=19,y=.96)
fig.text(.5,.915,'Schematic definitions for discussion — land is excluded from basin averages',ha='center',fontsize=11,color='#444444')
fig.text(.06,.12,'Atlantic: 70°W–20°E; north 0–90°N, south 0–60°S     |     Indian: 20°E–120°E, 60°S–30°N\nPacific: 120°E eastward to 70°W; north 0–90°N, south 0–60°S (split at map edge)\nSouthern: 60–90°S     |     No separate Arctic basin',fontsize=11,linespacing=1.6)
fig.text(.06,.035,'Rectangles are approximate domains, not geographic basin masks; marginal seas and passages require refinement.\nCoastlines: Natural Earth, 110m. No training basin definitions have been applied.',fontsize=9,color='#555555',linespacing=1.5)
fig.savefig(root/'proposed_basin_rectangles_split.png',dpi=220,facecolor='white')
print(root/'proposed_basin_rectangles_split.png')
