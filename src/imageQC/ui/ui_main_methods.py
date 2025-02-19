#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Methods for main window of imageQC.

Methods that can be reused in sub - automation processes e.g. task_based_image_quality.

@author: Ellen Wasbo
"""

# imageQC block start
from imageQC.scripts.calculate_roi import get_rois
from imageQC.scripts import dcm
from imageQC.scripts.artifact import apply_artifacts
# imageQC block end


def update_roi(main, clear_results_test=False):
    """Recalculate ROI."""
    errmsg = None
    if main.active_img is not None:
        main.start_wait_cursor()
        try:
            main.status_bar.showMessage('Updating ROI...')
        except AttributeError:
            pass
        try:
            main.current_roi, errmsg = get_rois(
                main.active_img,
                main.gui.active_img_no, main)
        except IndexError:  # might happen after closing images
            main.current_roi = None
        try:
            main.status_bar.clearMessage()
        except AttributeError:
            pass
    else:
        main.current_roi = None
    try:
        main.wid_image_display.canvas.roi_draw()
    except AttributeError:
        pass
    main.display_errmsg(errmsg)

    if main.current_test == 'SNI':
        try:
            if main.tab_nm.sni_selected_roi.count() == 2:
                main.tab_nm.update_sni_roi_names()
        except AttributeError:
            pass

    if clear_results_test:
        if main.current_test in [*main.results]:
            main.results[main.current_test] = None
            main.refresh_results_display()

    if main.active_img is not None:
        main.stop_wait_cursor()


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
                            main.current_test]['pr_image_sup'],
                        table_info=main.results[
                            main.current_test]['values_sup_info']
                        )
                except (KeyError, TypeError, IndexError):
                    main.wid_res_tbl_sup.result_table.clear()
        elif 'ResultPlotWidget' in type_wid:
            main.wid_res_plot.plotcanvas.plot()
        elif 'ResultImageWidget' in type_wid:
            main.wid_res_image.canvas.result_image_draw()


def refresh_img_display(main, force_home=False):
    """Refresh image related gui."""
    if main.active_img is not None:
        main.current_roi = None
        main.wid_image_display.canvas.img_draw(force_home=force_home)
        try:
            main.wid_window_level.canvas.plot(main.active_img)
            main.wid_dcm_header.refresh_img_info(
                main.imgs[main.gui.active_img_no].info_list_general,
                main.imgs[main.gui.active_img_no].info_list_modality)
        except AttributeError:
            pass
        except IndexError:  # maybe after closing images
            main.wid_image_display.canvas.img_is_missing()
        main.update_roi()
    else:
        main.wid_image_display.canvas.img_is_missing()


def sum_marked_images(main):
    """Sum marked images and apply artifacts if any."""
    errmsg = None
    if len(main.artifacts) == 0 and len(main.artifacts_3d) == 0:
        summed_img, errmsg = dcm.sum_marked_images(
            main.imgs, main.tree_file_list.get_marked_imgs_current_test(),
            tag_infos=main.tag_infos)
    else:
        marked = main.get_marked_imgs_current_test()
        summed_img = None
        for img_no, img_info in enumerate(main.imgs):
            if img_no in marked:
                arr, _ = dcm.get_img(
                    img_info.filepath,
                    frame_number=img_info.frame_number,
                    tag_infos=main.tag_infos, overlay=main.gui.show_overlay,
                    rotate_k=main.gui.rotate_k)
                if len(img_info.artifacts) > 0:
                    arr = apply_artifacts(
                        arr, img_info,
                        main.artifacts, main.artifacts_3d, img_no)
                if summed_img is None:
                    summed_img = arr
                else:
                    try:
                        summed_img = summed_img + arr
                    except ValueError:
                        errmsg = 'Failed summing images. Different dimensions?'
                        break
    return summed_img, errmsg

