import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import seaborn as sns
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import os
from matplotlib.dates import HourLocator, DateFormatter
import matplotlib.image as mpimg

import folium
import branca.colormap as cm
import io
import base64
from PIL import Image

from geojson import Feature, Point, FeatureCollection
import json
import matplotlib
import h3

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from models.modelutils import load_yaml_config

# parameter_dict = {
#     "air tem": "Air temperature",
#     "RH": "Relative humidity",
#     "wind speed": "Wind speed",
#     "solar radiation": "Global horizontal irradiance"
# }

parameter_dict = {
    "air tem": "$T_{air}$",
    "RH": "$RH$",
    "wind speed": "$v$",
    "solar radiation": "$GHI$"
}

unit_dict = {
    "air tem": "℃",
    "RH": "%",
    "wind speed": "m/s",
    "solar radiation": r"$\mathrm{W/m}^{2}$"
}

WS_dict = {
    "WS_02": "WS-1",
    "WS_03": "WS-2",
    "WS_04": "WS-3",
    "WS_06": "WS-4",
    "WS_07": "WS-5",
    "WS_08": "WS-6",
    "WS_11": "WS-7",
    "WS_13": "WS-8",
    "WS_16": "WS-9",
    "WS_17": "WS-10",
    "WS_18": "WS-11",
    "WS_20": "WS-12"
}

def get_weather_unit(weather_item):
    if weather_item == "air tem":
        weather_unit = "℃"
    elif weather_item == "RH":
        weather_unit = "%"
    elif weather_item == "AH":
        weather_unit = "g/m³"
    elif weather_item == "wind speed":
        weather_unit = "m/s"
    elif weather_item == "solar radiation":
        weather_unit = "W/m²"
    return weather_unit

def hexbin_plot_prediction(args, df, xymin=None, xymax=None, vmax=None):
    weather_item = args.target_weather
    weather_unit = get_weather_unit(weather_item)
    # density_scatter(df[weather_item+'_target_pred'], df[weather_item+'_target'], bins = [10,10] )
    # density_scatter(df[weather_item+'_reference'], df[weather_item+'_target'], bins = [10,10] )

    # Generate some random data
    np.random.seed(0)
    x = df[weather_item+'_target']
    y = df[weather_item+'_target_pred']

    xyrange = xymax-xymin
    x.loc[len(x)] = xymax + xyrange
    y.loc[len(y)] = xymax + xyrange
    x.loc[len(x)] = xymin - xyrange
    y.loc[len(y)] = xymin - xyrange

    # Create a figure and two subplots, side by side
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=(5, 4))

    #vmax = round(len(x)/50, -3)

    # First plot
    h = ax.hexbin(x, y, gridsize=100, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xlabel(f'{parameter_dict[weather_item]}, ground truth ({unit_dict[weather_item]})')
    ax.set_ylabel(f'{parameter_dict[weather_item]}, prediction ({unit_dict[weather_item]})')
    ax.set_xlim([xymin, xymax])
    ax.set_ylim([xymin, xymax])
    cb = fig.colorbar(h, ax=ax)
    cb.set_label('Counts')

    # Adding the x=y line to both subplots
    ax.plot([0, 1000], [0, 1000], color='white', linestyle='--')  # This plots the x=y line
    
    svg_file_path = os.path.join("figure", f"{args.target_weather}_prediction_hexbin.svg")
    plt.savefig(svg_file_path, format='svg')
    plt.show()

def hexbin_plot_baseline(args, df, xymin=None, xymax=None, vmax=None):

    weather_item = args.target_weather
    weather_unit = get_weather_unit(weather_item)
    # density_scatter(df[weather_item+'_target_pred'], df[weather_item+'_target'], bins = [10,10] )
    # density_scatter(df[weather_item+'_reference'], df[weather_item+'_target'], bins = [10,10] )

    # Generate some random data
    np.random.seed(0)
    x = df[weather_item+'_target']
    y = df[weather_item+'_reference']

    xyrange = xymax-xymin
    x.loc[len(x)] = xymax + xyrange
    y.loc[len(y)] = xymax + xyrange
    x.loc[len(x)] = xymin - xyrange
    y.loc[len(y)] = xymin - xyrange

    # Create a figure and two subplots, side by side
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=(5, 4))

    #vmax = round(len(x)/50, -3)

    # First plot
    h = ax.hexbin(x, y, gridsize=100, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xlabel(f'{parameter_dict[weather_item]}, target ({unit_dict[weather_item]})')
    ax.set_ylabel(f'{parameter_dict[weather_item]}, referemce ({unit_dict[weather_item]})')
    ax.set_xlim([xymin, xymax])
    ax.set_ylim([xymin, xymax])
    cb = fig.colorbar(h, ax=ax)
    cb.set_label('Counts')

    # Adding the x=y line to both subplots
    ax.plot([0, 1000], [0, 1000], color='white', linestyle='--')  # This plots the x=y line

    
    svg_file_path = os.path.join("figure", f"{args.target_weather}_baseline_hexbin.svg")
    plt.savefig(svg_file_path, format='svg')
    plt.show()


def WS_linechart(args, dates, start_date=None, y_min=None, y_max=None):

    weather_item = args.target_weather
    weather_unit = get_weather_unit(weather_item)

    # Load the data
    df = pd.read_csv(os.path.join(args.dataset_root, "microclimate", args.dataset_name_s+".csv"))

    # Convert 'time_reference' to datetime
    df['time'] = pd.to_datetime(df['time'])
    # df = df[df['time'].dt.date.isin(sunny_dates)]
    # df = df[~df['time'].dt.date.isin(rainy_dates)]
    df = df[df['time'].dt.date.isin(dates)]
    df = df[((df['time'].dt.hour >= 9) & (df['time'].dt.hour <= 17))]

    # Extract the unique dates after the start_date
    unique_dates = df[df['time'].dt.date > pd.to_datetime(start_date).date()]['time'].dt.date.unique()

    # Take only the next 3 unique dates
    days = 1

    dates_to_plot = unique_dates[:days]

    fig, axs = plt.subplots(1, len(dates_to_plot), figsize=(4, 2.5))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # You can add more colors if needed.
    linestyles = ['-', '--', '-.', ':']  # You can add more line styles if needed.

    for idx, date in enumerate(dates_to_plot):
        if days == 1:
            ax = axs
        else:
            ax = axs[idx]
        mask = (df['time'].dt.date == date)

        df_filtered = df.loc[mask]

        plots = []

        # Counter for color and linestyle
        color_counter = 0
        linestyle_counter = 0

        # iterate over unique WSname
        for ws_name in sorted(df_filtered['WSname'].unique(), key=lambda s: int(''.join(filter(str.isdigit, s.split(' ')[0])))):
            # create a mask for each Microclimate Weather Station
            ws_mask = df_filtered['WSname'] == ws_name
            df_ws_unique = df_filtered.loc[ws_mask, ['time', weather_item]].drop_duplicates()           
            
            #df_ws_unique['time_reference'] = pd.to_datetime(df_ws_unique['time_reference'])
            #df_ws_unique['time_target'] = pd.to_datetime(df_ws_unique['time_target'])
            #df_ws_unique = df_ws_unique[((df_ws_unique['time_reference'].dt.hour >= 7) & (df_ws_unique['time_reference'].dt.hour <= 19))]

            # Construct new label format (e.g., "WS-1" instead of "WS_02")
            new_label = WS_dict[ws_name]

            p, = ax.plot(df_ws_unique['time'], df_ws_unique[weather_item], label=new_label, color=colors[color_counter], linestyle=linestyles[linestyle_counter], alpha=0.7)
            plots.append(p)
            
            # Rotate through the colors and line styles
            color_counter += 1
            if color_counter >= len(colors):
                color_counter = 0
                linestyle_counter += 1
                if linestyle_counter >= len(linestyles):
                    linestyle_counter = 0  # reset linestyle_counter if we've used them all

        # Adjust x-axis to show only the hour
        ax.xaxis.set_major_locator(HourLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%H'))

        # ax.set_title(parameter_dict[weather_item] + ' Time Series for ' + str(date))
        ax.set_xlabel(str(date))
        ax.set_ylabel(parameter_dict[weather_item] + " " + f"({unit_dict[weather_item]})")
        ax.set_ylim([y_min, y_max])

        # After plotting your data, adjust y-axis limits
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin=ymin-(ymax-ymin)*0.1, ymax=ymax+(ymax-ymin)*0.1)  # Adjust the upper limit to 5% above the max value

        #ax.grid(True)
        ax.grid(True, linestyle=':', color='grey', linewidth=0.4)

        # Place the legend outside of the chart
        # ax.legend(handles=plots, loc='upper left', bbox_to_anchor=(1, 1))

        ax.tick_params(axis='both', which='both', length=0)  # No ticks
        for spine in ax.spines.values():  # Remove the outline
            spine.set_visible(False)

    # Save the figure as SVG before showing it
    output_path = f'figure/{weather_item}.svg'  # Change to a full path if needed
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()

def compare_WS_pano(df, weather_item, time):

    weather_unit = get_weather_unit(weather_item)

    # Convert 'time_reference' to datetime
    df = df[df['time'] == time]
    df = df.sort_values(by=weather_item, ascending=False).reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])

    fig, axes = plt.subplots(len(df), 1, figsize=(3, 12))
    axes = axes.ravel()

    for i in range(len(df)):
        ax = axes[i]
        WSname = df["WSname"][i]
        imgpath = f"data/raw/panorama/single/{WSname}_001.jpg"

        img = Image.open(imgpath)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        ax.set_title(os.path.basename(WSname))

    plt.tight_layout()
    plt.show()

def compare_WS_sate(df, weather_item, time):

    weather_unit = get_weather_unit(weather_item)

    # Convert 'time_reference' to datetime
    df = df[df['time'] == time]
    df = df.sort_values(by=weather_item, ascending=False).reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])

    fig, axes = plt.subplots(len(df), 1, figsize=(3, 12))
    axes = axes.ravel()

    for i in range(len(df)):
        ax = axes[i]
        WSname = df["WSname"][i]
        imgpath = f"data/raw/satellite/single/{WSname}_001.jpg"

        img = Image.open(imgpath)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        ax.set_title(os.path.basename(WSname))

    plt.tight_layout()
    plt.show()

def compare_WS_day_pano(df, weather_item, date):
    numWS = 12

    weather_unit = get_weather_unit(weather_item)

    # Convert 'time_reference' to datetime
    df['time'] = pd.to_datetime(df['time'])
    #print(df['time'].dt.date)
    df = df[df['time'].dt.date == pd.Timestamp(date).date()]
    
    #print(df)

    starthour = 6
    endhour = 20
    fig, axes = plt.subplots(numWS, endhour-starthour+1, figsize=(2*(endhour-starthour+1), 1.3*numWS))
    for j in range(starthour, endhour+1):
        df_temp = df[df['time'].dt.hour == j]
        df_temp = df_temp.sort_values(by=weather_item, ascending=False).reset_index(drop=True)
        #print(df_temp)
        for i in range(numWS):
            ax = axes[i][j-starthour]
            #print(df_temp)
            WSname = df_temp["WSname"][i]
            value = df_temp[weather_item][i]
            imgpath = f"data/raw/panorama/single/{WSname}_001.jpg"

            img = Image.open(imgpath)
            ax.imshow(img)
            ax.axis('off')  # Hide axes
            if i == 0:
                ax.set_title(f"{j}:00\n{WSname}: {value:.1f} {weather_unit}")
            else:
                ax.set_title(f"{WSname}: {value:.1f} {weather_unit}")

    plt.tight_layout()
    plt.show()

def compare_WS_day_sate(df, weather_item, date):
    numWS = 12

    weather_unit = get_weather_unit(weather_item)

    # Convert 'time_reference' to datetime
    df['time'] = pd.to_datetime(df['time'])
    #print(df['time'].dt.date)
    df = df[df['time'].dt.date == pd.Timestamp(date).date()]
    
    #print(df)

    starthour = 6
    endhour = 20
    fig, axes = plt.subplots(numWS, endhour-starthour+1, figsize=(2*(endhour-starthour+1), 2.3*numWS))
    for j in range(starthour, endhour+1):
        df_temp = df[df['time'].dt.hour == j]
        df_temp = df_temp.sort_values(by=weather_item, ascending=False).reset_index(drop=True)
        #print(df_temp)
        for i in range(numWS):
            ax = axes[i][j-starthour]
            #print(df_temp)
            WSname = df_temp["WSname"][i]
            value = df_temp[weather_item][i]
            imgpath = f"data/raw/satellite/single/{WSname}_001.jpg"

            img = Image.open(imgpath)
            ax.imshow(img)
            ax.axis('off')  # Hide axes
            if i == 0:
                ax.set_title(f"{j}:00\n{WSname}: {value:.1f} {weather_unit}")
            else:
                ax.set_title(f"{WSname}: {value:.1f} {weather_unit}")

    plt.tight_layout()
    plt.show()

def WS_linechart_oneref(args, dates, start_date=None, y_min=None, y_max=None):

    weather_item = args.target_weather
    weather_unit = get_weather_unit(weather_item)

    # Load the data
    df = pd.read_csv(os.path.join(args.dataset_root, "microclimate", args.dataset_name_s+".csv"))

    # Convert 'time_reference' to datetime
    df['time'] = pd.to_datetime(df['time'])
    # df = df[df['time'].dt.date.isin(sunny_dates)]
    # df = df[~df['time'].dt.date.isin(rainy_dates)]
    df = df[df['time'].dt.date.isin(dates)]
    df = df[((df['time'].dt.hour >= 9) & (df['time'].dt.hour <= 17))]

    # Extract the unique dates after the start_date
    unique_dates = df[df['time'].dt.date > pd.to_datetime(start_date).date()]['time'].dt.date.unique()

    # Take only the next 3 unique dates
    dates_to_plot = unique_dates[:3]

    fig, axs = plt.subplots(1, len(dates_to_plot), figsize=(16, 3))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # You can add more colors if needed.
    linestyles = ['-', '--', '-.', ':']  # You can add more line styles if needed.

    for idx, date in enumerate(dates_to_plot):
        ax = axs[idx]
        mask = (df['time'].dt.date == date)

        df_filtered = df.loc[mask]

        plots = []

        # Counter for color and linestyle
        color_counter = 0
        linestyle_counter = 0

        # iterate over unique WSname
        for ws_name in sorted(df_filtered['WSname'].unique(), key=lambda s: int(''.join(filter(str.isdigit, s.split(' ')[0])))):
            # create a mask for each Microclimate Weather Station
            ws_mask = df_filtered['WSname'] == ws_name
            df_ws_unique = df_filtered.loc[ws_mask, ['time', weather_item]].drop_duplicates()           
            
            #df_ws_unique['time_reference'] = pd.to_datetime(df_ws_unique['time_reference'])
            #df_ws_unique['time_target'] = pd.to_datetime(df_ws_unique['time_target'])
            #df_ws_unique = df_ws_unique[((df_ws_unique['time_reference'].dt.hour >= 7) & (df_ws_unique['time_reference'].dt.hour <= 19))]

            p, = ax.plot(df_ws_unique['time'], df_ws_unique[weather_item], label=ws_name, color=colors[color_counter], linestyle=linestyles[linestyle_counter])
            plots.append(p)
            
            # Rotate through the colors and line styles
            color_counter += 1
            if color_counter >= len(colors):
                color_counter = 0
                linestyle_counter += 1
                if linestyle_counter >= len(linestyles):
                    linestyle_counter = 0  # reset linestyle_counter if we've used them all

        # Adjust x-axis to show only the hour
        ax.xaxis.set_major_locator(HourLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%H'))

        ax.set_title(parameter_dict[weather_item] + ' Time Series for ' + str(date))
        ax.set_xlabel('Time')
        ax.set_ylabel(parameter_dict[weather_item] + " " + weather_unit)
        ax.set_ylim([y_min, y_max])
        ax.grid(True)

        # Place the legend outside of the chart
        ax.legend(handles=plots, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

def plot_time_series(args, config, ax, df, start_time, end_time):

    weather_item = args.target_weather
    weather_unit = get_weather_unit(weather_item)
    # Filter data by time and weather station
    if "panosate" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['panoIMGname_reference'] == config['panoIMGname_reference']) & (df['panoIMGname_target'] == config['panoIMGname_target']) & (df['sateIMGname_reference'] == config['sateIMGname_reference']) & (df['sateIMGname_target'] == config['sateIMGname_target'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'panoIMGname_reference'])        
    elif "pano" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['panoIMGname_reference'] == config['panoIMGname_reference']) & (df['panoIMGname_target'] == config['panoIMGname_target'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'panoIMGname_reference'])        
    elif "sate" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['sateIMGname_reference'] == config['sateIMGname_reference']) & (df['sateIMGname_target'] == config['sateIMGname_target'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'sateIMGname_reference'])    

    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_reference'], label=parameter_dict[weather_item]+'_reference', color='blue')
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_target'], label=parameter_dict[weather_item]+'_target', color='red')
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_target_pred'], label=parameter_dict[weather_item]+'_target_pred', color='red', linestyle='--') #,color='red', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel(weather_item+" "+f"({unit_dict[weather_item]})")
    ax.set_title(f'{start_time.strftime("%Y-%m-%d")}')
    ax.set_ylim([config["vmin"], config["vmax"]])

    # Adjust x-axis to show only the hour
    ax.xaxis.set_major_locator(HourLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%H'))

    ax.legend()
    ax.grid(True)

def plot_time_series_v2(args, config, ax, df, start_time, end_time):
    weather_item = args.target_weather
    weather_unit = get_weather_unit(weather_item)  # Assuming get_weather_unit is defined elsewhere

    # Generate a seaborn color palette
    palette = sns.color_palette("viridis", n_colors=10)
    colors = palette.as_hex()

    # Filter data by time and weather station
    if "panosate" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['panoIMGname_reference'] == config['panoIMGname_reference_1']) & (df['panoIMGname_target'] == config['panoIMGname_target_1']) & (df['sateIMGname_reference'] == config['sateIMGname_reference_1']) & (df['sateIMGname_target'] == config['sateIMGname_target_1'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'panoIMGname_reference'])
    elif "pano" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['panoIMGname_reference'] == config['panoIMGname_reference_1']) & (df['panoIMGname_target'] == config['panoIMGname_target_1'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'panoIMGname_reference'])
    elif "sate" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['sateIMGname_reference'] == config['sateIMGname_reference_1']) & (df['sateIMGname_target'] == config['sateIMGname_target_1'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'sateIMGname_reference'])

    # Plotting
    #ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_reference'], label=weather_item+'_reference', color=colors[0], alpha=0.6)
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_reference'], label=parameter_dict[weather_item]+'_reference', color="black", alpha=0.5)
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_target'], label=parameter_dict[weather_item]+'_target', color=colors[5])#, alpha=0.8)
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_target_pred'], label=parameter_dict[weather_item]+'_target_pred', color=colors[5], linestyle='--')#, alpha=0.8)

    # Filter data by time and weather station
    if "panosate" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['panoIMGname_reference'] == config['panoIMGname_reference_2']) & (df['panoIMGname_target'] == config['panoIMGname_target_2']) & (df['sateIMGname_reference'] == config['sateIMGname_reference_2']) & (df['sateIMGname_target'] == config['sateIMGname_target_2'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'panoIMGname_reference'])
    elif "pano" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['panoIMGname_reference'] == config['panoIMGname_reference_2']) & (df['panoIMGname_target'] == config['panoIMGname_target_2'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'panoIMGname_reference'])
    elif "sate" in args.model:
        filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['sateIMGname_reference'] == config['sateIMGname_reference_2']) & (df['sateIMGname_target'] == config['sateIMGname_target_2'])]
        filtered_df = filtered_df.drop_duplicates(subset=['time_reference', 'sateIMGname_reference'])

    # Plotting
    #ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_reference'], label=weather_item+'_reference', color=colors[0])
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_target'], label=parameter_dict[weather_item]+'_target', color=colors[9])#, alpha=0.8)
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_target_pred'], label=parameter_dict[weather_item]+'_target_pred', color=colors[9], linestyle='--')#, alpha=0.8)

    #ax.set_xlabel(f'{start_time.strftime("%m-%d-%Y")}')
    ax.set_xlabel(f'{start_time.strftime("%Y-%m-%d")}')
    ax.set_ylabel(parameter_dict[weather_item] + " " + f"({unit_dict[weather_item]})")
    #ax.set_title(f'{start_time.strftime("%Y-%m-%d")}')
    ax.set_ylim([config["vmin_line"], config["vmax_line"]])

    # Adjust x-axis to show only the hour
    ax.xaxis.set_major_locator(HourLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%H'))

    # Customize the grid and axes
    ax.grid(True, linestyle=':', color='grey', linewidth=0.4)#, axis='y')  # Dotted horizontal lines

    # After plotting your data, adjust y-axis limits
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=ymin-(ymax-ymin)*0.07, ymax=ymax+(ymax-ymin)*0.07)  # Adjust the upper limit to 5% above the max value

    # Add a horizontal line at 0
    # ax.axhline(0, color='grey', linestyle=':')

    ax.tick_params(axis='both', which='both', length=0)  # No ticks
    for spine in ax.spines.values():  # Remove the outline
        spine.set_visible(False)

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00))

def plot_time_series_oneref(args, ax, df, start_time, end_time, image_name_target, y_min, y_max):

    weather_item = args.target_weather
    weather_unit = get_weather_unit(weather_item)
    # Filter data by time and weather station
    filtered_df = df[(df['time_reference'] >= start_time) & (df['time_reference'] <= end_time) & (df['panoIMGname_target'] == image_name_target)]
    #filtered_df = filtered_df.drop_duplicates(subset=['time_reference'])

    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_reference'], label=parameter_dict[weather_item]+'_reference', color='blue')
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_target'], label=parameter_dict[weather_item]+'_target', color='red')
    ax.plot(filtered_df['time_reference'], filtered_df[weather_item+'_target_pred'], label=parameter_dict[weather_item]+'_target_pred', color='red', linestyle='--') #,color='red', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel(parameter_dict[weather_item]+" "+f"({unit_dict[weather_item]})")
    ax.set_title(f'{start_time.strftime("%Y-%m-%d")}')
    ax.set_ylim([y_min, y_max])

    # Adjust x-axis to show only the hour
    ax.xaxis.set_major_locator(HourLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%H'))

    ax.legend()
    ax.grid(True)

def display_prediction_timeseries(args, config, df):
    # Convert 'time_reference' column to datetime
    df['time_reference'] = pd.to_datetime(df['time_reference'])
    start_time = datetime.strptime(config["start_day"], "%Y-%m-%d")

    # Get unique dates after the start_time
    unique_dates = df[df['time_reference'] > start_time]['time_reference'].dt.date.unique()

    # Take only the first 3 dates
    days = 1
    dates_to_plot = unique_dates[:days]

    # Set up a 5-plot figure
    fig, axs = plt.subplots(1, len(dates_to_plot), figsize=(12, 3))

    for i, date in enumerate(dates_to_plot):
        start = pd.Timestamp(date)
        end = start + timedelta(days=1)
        plot_time_series(args, config, axs[i], df, start, end)

    plt.tight_layout()
    svg_file_path = os.path.join("figure", f"{args.target_weather}_prediction_line.svg")
    plt.savefig(svg_file_path, format='svg')
    plt.show()

def display_prediction_timeseries_v2(args, config, df):
    # Convert 'time_reference' column to datetime
    df['time_reference'] = pd.to_datetime(df['time_reference'])
    start_time = datetime.strptime(config["start_day"], "%Y-%m-%d")

    # Get unique dates after the start_time
    unique_dates = df[df['time_reference'] > start_time]['time_reference'].dt.date.unique()

    # Take only the first 3 dates
    days = 1
    dates_to_plot = unique_dates[:days]

    # Set up a 5-plot figure
    fig, axs = plt.subplots(1, len(dates_to_plot), figsize=(4, 2.5))

    for i, date in enumerate(dates_to_plot):
        start = pd.Timestamp(date)
        end = start + timedelta(days=1)
        if len(dates_to_plot) == 1:
            plot_time_series_v2(args, config, axs, df, start, end)
        else:
            plot_time_series_v2(args, config, axs[i], df, start, end)

    plt.tight_layout()
    svg_file_path = os.path.join("figure", f"{args.target_weather}_prediction_line.svg")
    plt.savefig(svg_file_path, format='svg')
    plt.show()

def display_prediction_timeseries_oneref(args, df, start_time, days, station_name_target, y_min=None, y_max=None):
    # Convert 'time_reference' column to datetime
    df['time_reference'] = pd.to_datetime(df['time_reference'])
    start_time = datetime.strptime(start_time, "%Y-%m-%d")

    # Get unique dates after the start_time
    unique_dates = df[df['time_reference'] > start_time]['time_reference'].dt.date.unique()

    # Take only the first 5 dates
    dates_to_plot = unique_dates[:days]

    # Set up a 5-plot figure
    fig, axs = plt.subplots(1, len(dates_to_plot), figsize=(12, 3))

    for i, date in enumerate(dates_to_plot):
        start = pd.Timestamp(date)
        end = start + timedelta(days=1)
        plot_time_series_oneref(args, axs[i], df, start, end, station_name_target, y_min, y_max)

    plt.tight_layout()
    plt.show()

def display_image_child(args, image_path_ref, image_path_tar):
    # Load the images
    img1 = mpimg.imread(image_path_ref)
    img2 = mpimg.imread(image_path_tar)
    title1 = 'Reference:' + os.path.basename(image_path_ref)
    title2 = 'Target:' + os.path.basename(image_path_tar)

    # Create a figure and a set of subplots
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 2))  # Adjust figsize as needed
    # Display the first image
    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis('off')  # Hide axes
    # Display the second image
    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()

def display_images(args, config):
    if "panosate" in args.model: 
        image_path_ref = os.path.join(args.pano_folder, config["panoIMGname_reference"])
        image_path_tar = os.path.join(args.pano_folder, config["panoIMGname_target"])
        display_image_child(args, image_path_ref, image_path_tar)  
        image_path_ref = os.path.join(args.sate_folder, config["sateIMGname_reference"])
        image_path_tar = os.path.join(args.sate_folder, config["sateIMGname_target"])
        display_image_child(args, image_path_ref, image_path_tar)   
    elif "pano" in args.model:   
        image_path_ref = os.path.join(args.pano_folder, config["panoIMGname_reference"])
        image_path_tar = os.path.join(args.pano_folder, config["panoIMGname_target"])
        display_image_child(args, image_path_ref, image_path_tar)         
    elif "sate" in args.model:  
        image_path_ref = os.path.join(args.sate_folder, config["sateIMGname_reference"])
        image_path_tar = os.path.join(args.sate_folder, config["sateIMGname_target"])
        display_image_child(args, image_path_ref, image_path_tar)

# Add custom basemaps to folium
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = True,
        control = True
    ),
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Google Terrain': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Terrain',
        overlay = True,
        control = True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
    ),
    'Light Map': folium.TileLayer(
        tiles = 'cartodbpositron',
        attr = 'cartodb',
        name = 'Light Map',
        overlay = True,
        control = True
    ),
    'Dark Map': folium.TileLayer(
        tiles = 'cartodbdark_matter',
        attr = 'cartodb',
        name = 'Dark Map',
        overlay = True,
        control = True
    )
}

def mapping_points_image_popup(df_map, value_name, vmin, vmax, imgdir):
    m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=18)

    folium.TileLayer('cartodbpositron',name="Light Map",control=False).add_to(m)
    folium.TileLayer('cartodbdark_matter',name="Dark Map",control=False).add_to(m)

    # Add custom basemaps
    basemaps['Google Maps'].add_to(m)
    basemaps['Light Map'].add_to(m)
    #basemaps['Dark Map'].add_to(m)
    #basemaps['Google Satellite Hybrid'].add_to(m)

    # create color range based on values
    #colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=df_map[item1].min(), vmax=df_map[item1].max())
    #colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0, vmax=df_map[item1].max())
    #colormap = folium.LinearColormap(colors=['blue', 'white', 'red'], vmin=31, vmax=35)
    #colormap = cm.linear.plasma.scale(31, 35)
    colormap = cm.linear.viridis.scale(vmin, vmax)
    # colormap = cm.LinearColormap(colors=['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'],
    #                              index=[30, 31, 32, 33, 34, 35], vmin=30, vmax=35,
    #                              caption='Total Standard deviation at the point[mm]')

    # add markers to map with color based on value range
    for i in range(len(df_map)):
        #for i in range(10):
        frame_path = os.path.join(imgdir, df_map["panoIMGname_target"][i])
        img = Image.open(frame_path)
        new_img = img.resize((256, 128))  # x, y
        buffer = io.BytesIO()
        new_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue())
        html = '<img src="data:image/png;base64,{}">'.format
        iframe = folium.IFrame(html(encoded.decode('UTF-8')), width=280, height=140)
        popup = folium.Popup(iframe, max_width=280)
        #html = '''frame_id:''' + str(df_map["frame_key"][i])
        #iframe = folium.IFrame(html, width=300, height=100)
        #popup = folium.Popup(iframe, max_width=300)
        folium.CircleMarker(location=[df_map.iloc[i]['lat'], df_map.iloc[i]['lon']], radius=1,
                                color=colormap(df_map.iloc[i][value_name]), fill=True, fill_opacity=1, popup=popup).add_to(m)

    m.add_child(colormap)

    #for coord in latlong:
    #  html = '''緯度:''' + str(coord[0]) + '''<br>''' + '''経度:''' + str(coord[1])
    #
    #  iframe = folium.IFrame(html, width=300, height=100)
    #  popup = folium.Popup(iframe, max_width=300)
    #  #folium.Marker( location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=1, popup=popup).add_to( mapit )
    #  folium.Circle(name='Emosca', location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=1, popup=popup).add_to( m )

    folium.LayerControl().add_to(m)

    return m


def hexagons_dataframe_to_geojson(df_hex, file_output = None, column_name = "value"):
    """
    Produce the GeoJSON for a dataframe, constructing the geometry from the "hex_id" column
    and with a property matching the one in column_name
    """    
    list_features = []
    
    for i,row in df_hex.iterrows():
        try:
            geometry_for_row = { "type" : "Polygon", "coordinates": [h3.h3_to_geo_boundary(h=row["hex_id"],geo_json=True)]}
            feature = Feature(geometry = geometry_for_row , id=row["hex_id"], properties = {column_name : row[column_name]})
            list_features.append(feature)
        except:
            print("An exception occurred for hex " + row["hex_id"]) 

    feat_collection = FeatureCollection(list_features)
    geojson_result = json.dumps(feat_collection)
    return geojson_result

def get_color(custom_cm, val, vmin, vmax):
    return matplotlib.colors.to_hex(custom_cm((val-vmin)/(vmax-vmin)))

def choropleth_map(df_aggreg, df_original, value_name, vmin, vmax, fill_opacity = 0.9, cmap = None, initial_map = None):
    """
    Creates choropleth maps given the aggregated data. initial_map can be an existing map to draw on top of.
    """    
    column_name = value_name

    #colormap
    min_value = df_aggreg[column_name].min()
    max_value = df_aggreg[column_name].max()
    mean_value = df_aggreg[column_name].mean()
    print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
    print(f"Hexagon cell count: {df_aggreg['hex_id'].nunique()}")
    
    # the name of the layer just needs to be unique, put something silly there for now:
    name_layer = "Choropleth " + str(df_aggreg)
    
    if initial_map is None:
        initial_map = folium.Map(location= [df_original['lat'].mean(), df_original['lon'].mean()], zoom_start=16, tiles="cartodbpositron")

    #create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg, column_name = column_name)

    # color_map_name 'Blues' for now, many more at https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose from!
    #colormap = matplotlib.cm.get_cmap(cmap)
    colormap = cm.LinearColormap(colors=plt.cm.plasma.colors, vmin=vmin, vmax=vmax)
    if cmap == "plasma":
        colormap = cm.LinearColormap(colors=plt.cm.plasma.colors, vmin=vmin, vmax=vmax)
    elif cmap == "inferno":
        colormap = cm.LinearColormap(colors=plt.cm.inferno.colors, vmin=vmin, vmax=vmax)
    elif cmap == "magma":
        colormap = cm.LinearColormap(colors=plt.cm.magma.colors, vmin=vmin, vmax=vmax)
    elif cmap == "Blues":
        num_samples = 256  # Number of samples to take from the colormap
        sample_points = [i/num_samples for i in range(num_samples)]
        colors = [plt.cm.Blues(point) for point in sample_points]
        # Creating the colormap
        colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    # elif cmap == "binary":
    #     colormap = cm.LinearColormap(colors=plt.cm.binary.colors, vmin=vmin, vmax=vmax)
    else:
        colormap = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=vmin, vmax=vmax)

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            #'fillColor': get_color(colormap, feature['properties'][column_name], vmin=vmin, vmax=vmax),
            'fillColor': colormap(feature['properties'][column_name]),
            #'color': border_color,
            'weight': 0,
            'fillOpacity': fill_opacity 
        }, 
        name = name_layer
    ).add_to(initial_map)

    # Add color scale as a caption to the map
    colormap.caption = value_name
    colormap.add_to(initial_map)

    return initial_map

def save_choropleth_svg(df_aggreg, df_original, value_name, vmin, vmax, fill_opacity = 0.7, cmap = None, initial_map = None, save_dir = None, save_name = None):
    """
    Creates choropleth maps given the aggregated data. initial_map can be an existing map to draw on top of.
    """    
    column_name = value_name

    #colormap
    min_value = df_aggreg[column_name].min()
    max_value = df_aggreg[column_name].max()
    mean_value = df_aggreg[column_name].mean()
    print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
    print(f"Hexagon cell count: {df_aggreg['hex_id'].nunique()}")
    
    # the name of the layer just needs to be unique, put something silly there for now:
    name_layer = "Choropleth " + str(df_aggreg)
    
    if initial_map is None:
        initial_map = folium.Map(location= [df_original['lat'].mean(), df_original['lon'].mean()], zoom_start=16, tiles="cartodbpositron")

    #create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg, column_name = column_name)
    geojson_data = json.loads(geojson_data)
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

    # Plotting the data with a color map based on the 'solar_radiation_target_pred' column
    #fig, ax = plt.subplots(facecolor='none')
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.patch.set_alpha(0.5)
    gdf.plot(column=column_name, cmap=cmap, ax=ax, legend=True, vmin=vmin, vmax=vmax)

    # Save the plot as an SVG file
    if save_name:
        user_colored_svg_file = f"{save_dir}/{value_name}_{save_name}_map.svg"
    else:
        user_colored_svg_file = f"{save_dir}/{value_name}_map.svg"

    #plt.savefig(user_colored_svg_file, format='svg', transparent=True, bbox_inches='tight')
    plt.savefig(user_colored_svg_file, format='svg')

    # color_map_name 'Blues' for now, many more at https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose from!
    #colormap = matplotlib.cm.get_cmap(cmap)

    # colormap = cm.LinearColormap(colors=plt.cm.plasma.colors, vmin=vmin, vmax=vmax)
    # if cmap == "plasma":
    #     colormap = cm.LinearColormap(colors=plt.cm.plasma.colors, vmin=vmin, vmax=vmax)
    # elif cmap == "inferno":
    #     colormap = cm.LinearColormap(colors=plt.cm.inferno.colors, vmin=vmin, vmax=vmax)
    # elif cmap == "magma":
    #     colormap = cm.LinearColormap(colors=plt.cm.magma.colors, vmin=vmin, vmax=vmax)
    # # elif cmap == "binary":
    # #     colormap = cm.LinearColormap(colors=plt.cm.binary.colors, vmin=vmin, vmax=vmax)
    # else:
    #     colormap = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=vmin, vmax=vmax)

def save_choropleth_with_basemap(df_aggreg, df_original, value_name, vmin, vmax, fill_opacity=0.7, cmap=None, initial_map=None, save_dir=None):
    """
    Creates choropleth maps with a raster basemap and expands the view area. 
    `expand_area_ratio` determines how much to expand the area around the data.
    """
    expand_area_ratio_x = 0.25
    expand_area_ratio_y = 0.15

    column_name = value_name

    # Create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex=df_aggreg, column_name=column_name)
    geojson_data = json.loads(geojson_data)
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

    # Set coordinate reference system to Web Mercator
    gdf = gdf.set_crs("EPSG:4326")  # Assuming your data is in WGS84
    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column=column_name, cmap=cmap, ax=ax, alpha=fill_opacity, legend=False, vmin=vmin, vmax=vmax)

    # Expand the area of the map
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) * expand_area_ratio_x
    dy = (maxy - miny) * expand_area_ratio_y
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)

    # Remove axis lines
    ax.set_axis_off()

    # Adding Contextily basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Save the plot as a PNG file
    user_colored_png_file = f"{save_dir}/{value_name}_map.png"
    plt.savefig(user_colored_png_file, format='png', bbox_inches='tight')

    plt.show()

def save_choropleth_with_basemap_time(df_aggreg, df_original, value_name, vmin, vmax, fill_opacity=0.7, cmap=None, initial_map=None, save_dir=None, save_name = None):
    """
    Creates choropleth maps with a raster basemap and expands the view area. 
    `expand_area_ratio` determines how much to expand the area around the data.
    """
    expand_area_ratio_x = 0.25
    expand_area_ratio_y = 0.15

    column_name = value_name

    # Create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex=df_aggreg, column_name=column_name)
    geojson_data = json.loads(geojson_data)
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

    # Set coordinate reference system to Web Mercator
    gdf = gdf.set_crs("EPSG:4326")  # Assuming your data is in WGS84
    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column=column_name, cmap=cmap, ax=ax, alpha=fill_opacity, legend=False, vmin=vmin, vmax=vmax)

    # Expand the area of the map
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) * expand_area_ratio_x
    dy = (maxy - miny) * expand_area_ratio_y
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)

    # Remove axis lines
    ax.set_axis_off()

    # Adding Contextily basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Save the plot as a PNG file
    user_colored_png_file = f"{save_dir}/{value_name}_map.png"

    # Save the plot as an SVG file
    if save_name:
        user_colored_png_file = f"{save_dir}/{value_name}_{save_name}_map.png"
    else:
        user_colored_png_file = f"{save_dir}/{value_name}_map.png"

    plt.savefig(user_colored_png_file, format='png', bbox_inches='tight')

    plt.show()

def mapping_h3_grid(df_map, value_name, vmin, vmax, resolution = None, cmap = None, save_dir = None, fill_opacity = None):
    hex_ids = df_map.apply(lambda row: h3.geo_to_h3(row.lat, row.lon, resolution), axis = 1)
    df_map = df_map.assign(hex_id=hex_ids.values)
    df_h3 = df_map.groupby("hex_id", as_index=False).agg({value_name: "mean"})
    save_choropleth_svg(df_h3, df_map, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir)
    save_choropleth_with_basemap(df_h3, df_map, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir)
    return choropleth_map(df_h3, df_map, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None)

def mapping_h3_grid_timeseries(df_map, value_name, vmin, vmax, resolution = None, cmap = None, save_dir = None, fill_opacity = None):
    df_map['time_reference'] = pd.to_datetime(df_map['time_reference'])
    times = list(set(df_map["time_reference"].tolist()))

    for time in times:
        # Convert to datetime object
        # date_obj = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        formatted_date_str = time.strftime('%Y-%m-%d-%H')

        df_map_filtered = df_map[df_map['time_reference'] == time]
        hex_ids = df_map_filtered.apply(lambda row: h3.geo_to_h3(row.lat, row.lon, resolution), axis = 1)
        df_map_filtered = df_map_filtered.assign(hex_id=hex_ids.values)
        df_h3 = df_map_filtered.groupby("hex_id", as_index=False).agg({value_name: "mean"})

        save_choropleth_svg(df_h3, df_map_filtered, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir, save_name = formatted_date_str)
        save_choropleth_with_basemap_time(df_h3, df_map_filtered, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir, save_name = formatted_date_str)

def mapping_h3_grid_timeseries_normalized(df_map, value_name, resolution = None, cmap = None, save_dir = None, fill_opacity = None):
    df_map['time_reference'] = pd.to_datetime(df_map['time_reference'])
    times = list(set(df_map["time_reference"].tolist()))

    for time in times:
        # Convert to datetime object
        # date_obj = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        formatted_date_str = time.strftime('%Y-%m-%d-%H')

        df_map_filtered = df_map[df_map['time_reference'] == time]
        hex_ids = df_map_filtered.apply(lambda row: h3.geo_to_h3(row.lat, row.lon, resolution), axis = 1)
        df_map_filtered = df_map_filtered.assign(hex_id=hex_ids.values)
        df_h3 = df_map_filtered.groupby("hex_id", as_index=False).agg({value_name: "mean"})

        vmax = df_h3[value_name].max()
        vmin = df_h3[value_name].min()

        print(time, vmin, vmax)

        save_choropleth_svg(df_h3, df_map_filtered, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir, save_name = formatted_date_str)
        save_choropleth_with_basemap_time(df_h3, df_map_filtered, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir, save_name = formatted_date_str)

def test_vis(config):
    # Set the style to 'ggplot'
    # plt.style.use('seaborn-v0_8-deep')
    test_dir_path = config["test_dir_path"]
    xymin, xymax = config["vmin"], config["vmax"]

    args = load_yaml_config(os.path.join(config["test_dir_path"], "hparams.yaml"))
    csv_files = [file for file in os.listdir(config["test_dir_path"]) if file.endswith(".csv")]

    df = pd.read_csv(os.path.join(test_dir_path, csv_files[0]))

    # calculate statistics
    mse = mean_squared_error(df[args.target_weather+'_target'], df[args.target_weather+'_target_pred'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df[args.target_weather+'_target'], df[args.target_weather+'_target_pred'])
    r2 = r2_score(df[args.target_weather+'_target'], df[args.target_weather+'_target_pred'])
    print(f'prediction and ground truth')
    print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}')
    hexbin_plot_prediction(args, df, xymin=xymin, xymax=xymax, vmax=150)

    mse = mean_squared_error(df[args.target_weather+'_target'], df[args.target_weather+'_reference'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df[args.target_weather+'_target'], df[args.target_weather+'_reference'])
    r2 = r2_score(df[args.target_weather+'_target'], df[args.target_weather+'_reference'])
    print(f'reference and ground truth')
    print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}')
    hexbin_plot_baseline(args, df, xymin=xymin, xymax=xymax, vmax=150)

    df = pd.read_csv(os.path.join(test_dir_path, csv_files[1]))
    start_time = config['start_day']

    if "panosate" in args.model:
        config["panoIMGname_reference"] =  f'pano_{config["ref_WSname_1"]}_{config["ref_pano_suffix_1"]}.jpg'
        config["panoIMGname_target"] =  f'pano_{args.dropwss[0]}_{config["tar_pano_suffix_1"]}.jpg'
        config["sateIMGname_reference"] = f'sate_{config["ref_WSname_1"]}_{config["ref_sate_suffix_1"]}.jpg'
        config["sateIMGname_target"]  = f'sate_{args.dropwss[0]}_{config["tar_sate_suffix_1"]}.jpg'
        display_images(args, config)
        display_prediction_timeseries(args, config, df)
        #start_time = '2019-02-01'
        config["panoIMGname_reference"] = f'pano_{config["ref_WSname_2"]}_{config["ref_pano_suffix_2"]}.jpg'
        config["panoIMGname_target"] = f'pano_{args.dropwss[1]}_{config["tar_pano_suffix_2"]}.jpg'
        config["sateIMGname_reference"] = f'sate_{config["ref_WSname_2"]}_{config["ref_sate_suffix_2"]}.jpg'
        config["sateIMGname_target"] = f'sate_{args.dropwss[1]}_{config["tar_sate_suffix_2"]}.jpg'
        display_images(args, config)
        display_prediction_timeseries(args, config, df)        
    elif "pano" in args.model:
        config["panoIMGname_reference"] =  f'pano_{config["ref_WSname_1"]}_{config["ref_pano_suffix_1"]}.jpg'
        config["panoIMGname_target"] =  f'pano_{args.dropwss[0]}_{config["tar_pano_suffix_1"]}.jpg'
        display_images(args, config)
        display_prediction_timeseries(args, config, df)
        #start_time = '2019-02-01'
        config["panoIMGname_reference"] = f'pano_{config["ref_WSname_2"]}_{config["ref_pano_suffix_2"]}.jpg'
        config["panoIMGname_target"] = f'pano_{args.dropwss[1]}_{config["tar_pano_suffix_2"]}.jpg'
        display_images(args, config)
        display_prediction_timeseries(args, config, df)       
    elif "sate" in args.model:
        config["sateIMGname_reference"] = f'sate_{config["ref_WSname_1"]}_{config["ref_sate_suffix_1"]}.jpg'
        config["sateIMGname_target"]  = f'sate_{args.dropwss[0]}_{config["tar_sate_suffix_1"]}.jpg'
        display_images(args, config)
        display_prediction_timeseries(args, config, df)
        #start_time = '2019-02-01'
        config["sateIMGname_reference"] = f'sate_{config["ref_WSname_2"]}_{config["ref_sate_suffix_2"]}.jpg'
        config["sateIMGname_target"] = f'sate_{args.dropwss[1]}_{config["tar_sate_suffix_2"]}.jpg'
        display_images(args, config)
        display_prediction_timeseries(args, config, df)

def test_vis_v2(config):
    # Set the style to 'ggplot'
    # plt.style.use('seaborn-v0_8-deep')
    test_dir_path = config["test_dir_path"]
    xymin, xymax = config["vmin"], config["vmax"]

    args = load_yaml_config(os.path.join(config["test_dir_path"], "hparams.yaml"))
    csv_files = [file for file in os.listdir(config["test_dir_path"]) if file.endswith(".csv")]

    df = pd.read_csv(os.path.join(test_dir_path, csv_files[0]))

    # calculate statistics
    mse = mean_squared_error(df[args.target_weather+'_target'], df[args.target_weather+'_target_pred'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df[args.target_weather+'_target'], df[args.target_weather+'_target_pred'])
    r2 = r2_score(df[args.target_weather+'_target'], df[args.target_weather+'_target_pred'])
    print(f'prediction and ground truth')
    print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}')
    hexbin_plot_prediction(args, df, xymin=xymin, xymax=xymax, vmax=150)

    mse = mean_squared_error(df[args.target_weather+'_target'], df[args.target_weather+'_reference'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df[args.target_weather+'_target'], df[args.target_weather+'_reference'])
    r2 = r2_score(df[args.target_weather+'_target'], df[args.target_weather+'_reference'])
    print(f'reference and ground truth')
    print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}')
    hexbin_plot_baseline(args, df, xymin=xymin, xymax=xymax, vmax=150)

    df = pd.read_csv(os.path.join(test_dir_path, csv_files[1]))
    start_time = config['start_day']

    if "panosate" in args.model:
        config["panoIMGname_reference_1"] =  f'pano_{config["ref_WSname_1"]}_{config["ref_pano_suffix_1"]}.jpg'
        config["panoIMGname_target_1"] =  f'pano_{args.dropwss[0]}_{config["tar_pano_suffix_1"]}.jpg'
        config["sateIMGname_reference_1"] = f'sate_{config["ref_WSname_1"]}_{config["ref_sate_suffix_1"]}.jpg'
        config["sateIMGname_target_1"]  = f'sate_{args.dropwss[0]}_{config["tar_sate_suffix_1"]}.jpg'
        #display_images(args, config)
        #display_prediction_timeseries(args, config, df)
        #start_time = '2019-02-01'
        config["panoIMGname_reference_2"] = f'pano_{config["ref_WSname_2"]}_{config["ref_pano_suffix_2"]}.jpg'
        config["panoIMGname_target_2"] = f'pano_{args.dropwss[1]}_{config["tar_pano_suffix_2"]}.jpg'
        config["sateIMGname_reference_2"] = f'sate_{config["ref_WSname_2"]}_{config["ref_sate_suffix_2"]}.jpg'
        config["sateIMGname_target_2"] = f'sate_{args.dropwss[1]}_{config["tar_sate_suffix_2"]}.jpg'
        #display_images(args, config)
        display_prediction_timeseries_v2(args, config, df)

    elif "pano" in args.model:
        config["panoIMGname_reference_1"] =  f'pano_{config["ref_WSname_1"]}_{config["ref_pano_suffix_1"]}.jpg'
        config["panoIMGname_target_1"] =  f'pano_{args.dropwss[0]}_{config["tar_pano_suffix_1"]}.jpg'
        # display_images(args, config)
        # display_prediction_timeseries(args, config, df)
        #start_time = '2019-02-01'
        config["panoIMGname_reference_2"] = f'pano_{config["ref_WSname_2"]}_{config["ref_pano_suffix_2"]}.jpg'
        config["panoIMGname_target_2"] = f'pano_{args.dropwss[1]}_{config["tar_pano_suffix_2"]}.jpg'
        # display_images(args, config)
        display_prediction_timeseries_v2(args, config, df)       
    elif "sate" in args.model:
        config["sateIMGname_reference_1"] = f'sate_{config["ref_WSname_1"]}_{config["ref_sate_suffix_1"]}.jpg'
        config["sateIMGname_target_1"]  = f'sate_{args.dropwss[0]}_{config["tar_sate_suffix_1"]}.jpg'
        # display_images(args, config)
        # display_prediction_timeseries(args, config, df)
        #start_time = '2019-02-01'
        config["sateIMGname_reference_2"] = f'sate_{config["ref_WSname_2"]}_{config["ref_sate_suffix_2"]}.jpg'
        config["sateIMGname_target_2"] = f'sate_{args.dropwss[1]}_{config["tar_sate_suffix_2"]}.jpg'
        # display_images(args, config)
        display_prediction_timeseries_v2(args, config, df)