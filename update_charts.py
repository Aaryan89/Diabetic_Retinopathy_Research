import os
from pptx import Presentation
from pptx.chart.data import CategoryChartData

prs = Presentation('Diabetic_Retinopathy_Presentation.pptx')
slide = prs.slides[11]

# Assuming two charts on slide 11
chart_qwk = slide.shapes[0].chart if slide.shapes[0].has_chart else (slide.shapes[1].chart if slide.shapes[1].has_chart else None)
chart_err = slide.shapes[1].chart if slide.shapes[1].has_chart else (slide.shapes[0].chart if slide.shapes[0].has_chart else slide.shapes[2].chart)

# Update QWK Chart
for shape in slide.shapes:
    if shape.has_chart:
        chart = shape.chart
        series_name = chart.series[0].name
        
        chart_data = CategoryChartData()
        chart_data.categories = ['A Softmax', 'B CORAL', 'C Regression', 'D CORN']
        
        if 'QWK' in series_name:
            chart_data.add_series('QWK (higher is better)', (0.8724, 0.7273, 0.8788, 0.8482))
        else:
            # Assume Severe Error >= 2
            chart_data.add_series(series_name, (5.6, 26.4, 3.8, 5.1))
        
        chart.replace_data(chart_data)

prs.save('Diabetic_Retinopathy_Presentation.pptx')
