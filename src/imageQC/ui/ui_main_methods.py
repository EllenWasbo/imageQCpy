#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Methods for main window of imageQC.

Methods that can be reused in sub - automation processes e.g. task_based_image_quality.

@author: Ellen Wasbo
"""

# imageQC block start
from imageQC.scripts.calculate_roi import get_rois
# imageQC block end


def update_roi(main, clear_results_test=False):
    """Recalculate ROI."""
    errmsg = None
    if main.active_img is not None:
        main.start_wait_cursor()
        main.status_bar.showMessage('Updating ROI...')
        main.current_roi, errmsg = get_rois(
            main.active_img,
            main.gui.active_img_no, main)
        main.status_bar.clearMessage()
        main.stop_wait_cursor()
    else:
        main.current_roi = None
    main.wid_image_display.canvas.roi_draw()
    main.display_errmsg(errmsg)
    if clear_results_test:
        if main.current_test in [*main.results]:
            main.results[main.current_test] = None
            main.refresh_results_display()


def refresh_results_display(main, update_table=True):
    """Update GUI for test results when results or selections change."""
    if main.current_test not in main.results:
        # clear all
        main.wid_res_tbl.result_table.clear()
        main.wid_res_tbl_sup.result_table.clear()
        main.wid_res_plot.plotcanvas.plot()
        main.wid_res_image.canvas.result_image_draw()
    else:
        # update only active
        wid = main.tab_results.currentWidget()
        type_wid = str(type(wid))
        if 'ResultTableWidget' in type_wid and update_table:
            if main.current_test == 'vendor':
                main.wid_res_tbl.result_table.fill_table(vendor=True)
            else:
                try:
                    main.wid_res_tbl.result_table.fill_table(
                        col_labels=main.results[main.current_test]['headers'],
                        values_rows=main.results[main.current_test]['values'],
                        linked_image_list=main.results[
                            main.current_test]['pr_image'],
                        table_info=main.results[main.current_test]['values_info']
                        )
                except (KeyError, TypeError, IndexError):
                    main.wid_res_tbl.result_table.clear()
                try:
                    main.wid_res_tbl_sup.result_table.fill_table(
                        col_labels=main.results[main.current_test]['headers_sup'],
                        values_rows=main.results[main.current_test]['values_sup'],
                        linked_image_list=main.results[
                            main.current_test]['pr_image'],
                        table_info=main.results[
                            main.current_test]['values_sup_info']
                        )
                except (KeyError, TypeError, IndexError):
                    main.wid_res_tbl_sup.result_table.clear()
        elif 'ResultPlotWidget' in type_wid:
            main.wid_res_plot.plotcanvas.plot()
        elif 'ResultImageWidget' in type_wid:
            main.wid_res_image.canvas.result_image_draw()


def refresh_img_display(main):
    """Refresh image related gui."""
    if main.active_img is not None:
        main.current_roi = None
        main.wid_image_display.canvas.img_draw()
        main.wid_window_level.canvas.plot(main.active_img)
        main.wid_dcm_header.refresh_img_info(
            main.imgs[main.gui.active_img_no].info_list_general,
            main.imgs[main.gui.active_img_no].info_list_modality)
        main.update_roi()
    else:
        main.wid_image_display.canvas.img_is_missing()
