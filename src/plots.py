import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import seaborn as sb
import cleanup as c

df = c.df

sea = sb.FacetGrid(df, col = "top genre", col_wrap=3, height = 4)
figa = sea.map_dataframe(sb.lineplot, "top year", "tempo")
figa.savefig("img/tempo.png")

sea = sb.FacetGrid(df, col = "top genre", col_wrap=3, height = 4)
figb = sea.map_dataframe(sb.lineplot, "top year", "popularity")
figb.savefig("img/popularity.png")

sea = sb.FacetGrid(df, col = "top genre", col_wrap=3, height = 4)
figc = sea.map_dataframe(sb.lineplot, "top year", "speechiness")
figc.savefig("img/speechiness.png")

sea = sb.FacetGrid(df, col = "top genre", col_wrap=3, height = 4)
figd = sea.map_dataframe(sb.lineplot, "top year", "danceability")
figd.savefig("img/danceability.png")
