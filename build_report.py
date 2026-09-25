#!/usr/bin/env python3
"""Build the PDF only from checked detector output and a real GitHub code URL."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)


ROOT = Path(__file__).resolve().parent
FIELDS = ('ClassID', 'Confidence', 'Left', 'Top', 'Right', 'Bottom',
          'Width', 'Height', 'Area', 'Center')
BLUE = colors.HexColor('#173b60')
PALE = colors.HexColor('#eef3f8')
GREEN = colors.HexColor('#267b50')


def fmt(field, value):
    if field == 'ClassID':
        return str(value)
    if field == 'Center':
        return '({:.4f}, {:.4f}) px'.format(*value)
    if field == 'Confidence':
        return '{:.6f}  ({:.2f}%)'.format(value, value * 100)
    return '{:.4f} {}'.format(value, 'px2' if field == 'Area' else 'px')


def make_report(results_file, github_url, output_file):
    if not github_url.startswith('https://github.com/') or not github_url.endswith('/your-detection.py'):
        raise ValueError('Provide the actual GitHub URL ending in /your-detection.py')
    data = json.loads(results_file.read_text(encoding='utf-8'))
    if not data.get('complete') or len(data['images']) != 2:
        raise ValueError('The actual two-image run is incomplete')
    for item in data['images']:
        if not item.get('selected_detection'):
            raise ValueError('A selected detection is missing')
        if any(field not in item['selected_detection'] for field in FIELDS):
            raise ValueError('At least one required detection field is missing')
        if not (results_file.parent / item['annotated']).is_file():
            raise ValueError('Annotated image is missing: ' + item['annotated'])

    styles = getSampleStyleSheet()
    title = ParagraphStyle('TitleCustom', parent=styles['Title'], fontName='Helvetica-Bold',
                           fontSize=20, leading=24, textColor=BLUE, spaceAfter=5)
    heading = ParagraphStyle('HeadingCustom', parent=styles['Heading2'], fontName='Helvetica-Bold',
                             fontSize=12, leading=15, textColor=BLUE, spaceBefore=9, spaceAfter=5)
    body = ParagraphStyle('BodyCustom', parent=styles['BodyText'], fontName='Helvetica',
                          fontSize=8.6, leading=12, spaceAfter=5)
    small = ParagraphStyle('SmallCustom', parent=body, fontSize=7.5, leading=10)
    caption = ParagraphStyle('CaptionCustom', parent=body, alignment=TA_CENTER,
                             textColor=GREEN, fontName='Helvetica-Bold', fontSize=8.2)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_file), pagesize=A4,
                            leftMargin=42, rightMargin=42,
                            topMargin=35, bottomMargin=38)
    story = []
    link = escape(github_url)
    story.append(Paragraph('ELEC7023 | Assignment 3', title))
    story.append(Paragraph('Two-image object detection with SSD MobileNet v2', body))
    story.append(Paragraph('<b>Execution environment:</b> macOS CPU, OpenCV {}. '
                           'This is an adaptation of Lecture 4\'s Jetson/TensorRT exercise; '
                           'the results below were generated on a Mac, not on a Jetson.'.format(
                               escape(data['runtime'].split(' ')[1])), body))
    story.append(Paragraph('<b>Model:</b> {} | confidence threshold {:.2f} | '
                           'selected class: {}'.format(
                               escape(data['model']), data['threshold'], escape(data['selected_class'])), body))
    story.append(Paragraph('<b>Code:</b> <link href="{}" color="blue">{}</link>'.format(link, link), small))
    story.append(Paragraph('<b>Image source:</b> official jetson-inference sample images '
                           '(<link href="https://github.com/dusty-nv/jetson-inference/tree/master/data/images" '
                           'color="blue">GitHub source folder</link>). Input photographs are '
                           'upstream examples, not photographs taken by the student.', small))
    story.append(Paragraph('Method', heading))
    story.append(Paragraph('The TensorFlow SSD MobileNet v2 COCO model was loaded with '
                           'OpenCV DNN. Each image was processed independently at its original '
                           'resolution and a 300 x 300 model input size. All detections scoring at least '
                           '0.50 were recorded. For each image, the highest-confidence dog '
                           'was selected and its bounding box is drawn in green. Other detections '
                           'remain in results.json. Class IDs follow the official COCO label map.', body))
    story.append(Paragraph('Result at a glance', heading))
    summary = [['Input image', 'Selected class', 'ClassID', 'Confidence']]
    for item in data['images']:
        selected = item['selected_detection']
        summary.append([item['input'], selected['ClassName'], str(selected['ClassID']),
                        '{:.2f}%'.format(selected['Confidence'] * 100)])
    summary_table = Table(summary, colWidths=[135, 115, 70, 105], hAlign='CENTER',
                          rowHeights=[24, 26, 26])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    thumbs = []
    for item in data['images']:
        img_path = results_file.parent / item['annotated']
        iw, ih = ImageReader(str(img_path)).getSize()
        thumbs.append(Image(str(img_path), width=205, height=205 * ih / iw))
    thumb_table = Table([[thumbs[0], thumbs[1]]], colWidths=[215, 215], hAlign='CENTER')
    story.append(thumb_table)
    story.append(Paragraph('The following pages present each annotated image at larger size '
                           'and all ten required output fields.', caption))

    for image_index, item in enumerate(data['images'], 1):
        story.append(PageBreak())
        story.append(Paragraph('Image {} | {}'.format(image_index, escape(item['input'])), heading))
        img_path = results_file.parent / item['annotated']
        iw, ih = ImageReader(str(img_path)).getSize()
        target_w = 400
        story.append(Image(str(img_path), width=target_w, height=target_w * ih / iw))
        story.append(Spacer(1, 4))
        selected = item['selected_detection']
        story.append(Paragraph('Selected detection: {} (ClassID {}) | original image {} x {} px'.format(
            escape(selected['ClassName']), selected['ClassID'], item['width'], item['height']), caption))
        rows = [['Required field', 'Actual detector result']]
        for field in FIELDS:
            rows.append([field, fmt(field, selected[field])])
        table = Table(rows, colWidths=[145, 280], hAlign='CENTER', rowHeights=[19] + [18] * 10)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 9),
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, BLUE),
        ]))
        story.append(table)
        if image_index == 2:
            story.append(Paragraph('Interpretation and provenance', heading))
            story.append(Paragraph(
                'Pixel coordinates use the upper-left corner as (0, 0). '
                'Width = Right - Left; Height = Bottom - Top; Area = Width x Height '
                '(square pixels); Center = ((Left + Right)/2, (Top + Bottom)/2). '
                'Coordinates are clipped to the image edge before measuring. '
                'Confidence is a model score, not an independently measured accuracy. '
                'The bounding-box area is not the object silhouette area.', small))
            story.append(Paragraph(
                'Sources: Lecture 4 (pp. 21-34); '
                '<link href="https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API" '
                'color="blue">OpenCV TensorFlow Object Detection API guide</link>; '
                '<link href="https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt" '
                'color="blue">TensorFlow COCO label map</link>. '
                'Full precision and SHA-256 hashes are in results.json.', small))

    def footer(canvas, document):
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(BLUE)
        canvas.drawString(42, 22, 'ELEC7023 | Assignment 3 | macOS OpenCV execution')
        canvas.drawRightString(A4[0] - 42, 22, 'Page {}'.format(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--github-url', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'Assignment3_Report.pdf')
    args = parser.parse_args()
    print(make_report(args.results, args.github_url, args.output))


if __name__ == '__main__':
    main()
