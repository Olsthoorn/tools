# tools

Various tools developed to do geo-hydrological work and modeling.

Note that the tools were developed as needed. They will be adapted as needed in the future without notice. These tools also reflect my improving skills in python, which also naturally leads to adaptations. There's neither any warranty nor guarantee, but you can be sure that I use them all the time, and that at the moment I use them I will make sure they work. But of course, when I change code, this might break other code using it, although I try to prevent this. This is merely a consequence of a developing environment, which was intended for myself.

Any suggestions are welcome.

Theo Olsthoorn2018/1/25


colors.py:  legal colornames such that you can used them in plots with multiple lines and can guarantee what color is used.
         All legal color names except those containgin "white" and sampled every 7 to ensure firmly distinct successive colors.

coords:  coordinate conversion, especialy Dutch nationa to and from WGS84 of Google.

dino:  dealing with information from www.dinoloket.nl

diver:  handling groundwater times series registered by the "Van Essen/Schlumberger" diver pressure logger.

flopy_fixes: .bugs in flopy, encountered and fixed

fdm:  finite difference modeling

googlemaps:  fetching a map from googlemaps on given coordinates, like that of a pump site

miscellaneous :  what is says, various things unordered.mlu  = pumping test analyse, handling files produced with the package
mlu (www.microfem.com)

kml:  producing and parsing klm files (Google Earth)

KNMI:  handling archived weather data from KNMI (Dutch National Weather Institute)

shape:  handling shapefiles

tsa:  time-series analysis

xml:  producing and parsing xml files

pra.py:  print a 2D numpy ndarray the matlab way (in one line).
