# (c) Copyright by Pierre-Henri Wuillemin, UPMC, 2017
# (pierre-henri.wuillemin@lip6.fr)
# Permission to use, copy, modify, and distribute this
# software and its documentation for any purpose and
# without fee or royalty is hereby granted, provided
# that the above copyright notice appear in all copies
# and that both that copyright notice and this permission
# notice appear in supporting documentation or portions
# thereof, including modifications, that you make.
# THE AUTHOR P.H. WUILLEMIN  DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
# SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOFTWARE!

"""
tools for exporting model and infernece as image
"""
import os
import re
import tempfile

import matplotlib.image as mpimg
import pyAgrum as gum
import pydot as dot
from pyAgrum.lib.bn2graph import BN2dot, BNinference2dot
from pyAgrum.lib.cn2graph import CN2dot, CNinference2dot
from pyAgrum.lib.id2graph import ID2dot, LIMIDinference2dot
from pyAgrum.lib.mn2graph import (
    MN2FactorGraphdot,
    MN2UGdot,
    MNinference2FactorGraphdot,
    MNinference2UGdot,
)


def export(model, filename=None, **kwargs):
    """
    export the graphical representation of the model in filename (png, pdf,etc.)

    :param GraphicalModel model: the model to show (pyAgrum.BayesNet, pyAgrum.MarkovNet, pyAgrum.InfluenceDiagram or pyAgrum.Potential)
    :param str filename: the name of the resulting file (suffix in ['pdf', 'png', 'fig', 'jpg', 'svg', 'ps']). If filename is None, the result is a np.array ready to be used with imshow().

    .. warning::
      Model can also just possess a method `toDot()` or even be a simple string in dot syntax.
    """
    if filename is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        export(model, tmp.name, **kwargs)
        img = mpimg.imread(tmp.name)
        try:
            os.remove(tmp.name)
        except (
            PermissionError
        ):  # probably windows error : file still 'used' ... grrr...
            pass
        return img

    fmt_image = filename.split(".")[-1]
    if fmt_image not in ["pdf", "png", "fig", "jpg", "svg", "ps"]:
        raise Exception(
            f"{filename} in not a correct filename for export : extension '{fmt_image}' not in [pdf,png,fig,jpg,svg]."
        )

    if isinstance(model, gum.BayesNet):
        fig = BN2dot(model, **kwargs)
    elif isinstance(model, gum.MarkovNet):
        if gum.config["notebook", "default_markovnetwork_view"] == "graph":
            fig = MN2UGdot(model, **kwargs)
        else:
            fig = MN2FactorGraphdot(model, **kwargs)
    elif isinstance(model, gum.InfluenceDiagram):
        fig = ID2dot(model, **kwargs)
    elif isinstance(model, gum.CredalNet):
        fig = CN2dot(model, **kwargs)
    elif isinstance(model, dot.Dot):
        fig = model
    elif hasattr(model, "toDot"):
        fig = dot.graph_from_dot_data(model.toDot())[0]

        # workaround for some badly parsed graph (pyparsing>=3.03)
        fig.del_node('"\\n"')
        fig.del_node('"\\n\\n"')
    elif isinstance(model, str):
        fig = dot.graph_from_dot_data(model)[0]

        # workaround for some badly parsed graph (pyparsing>=3.03)
        fig.del_node('"\\n"')
        fig.del_node('"\\n\\n"')
    else:
        raise gum.InvalidArgument(
            "Argument model should be a PGM (BayesNet, MarkovNet or Influence Diagram) or has a method `toDot()` or is a string"
        )
    fig.write(filename, format=fmt_image)


def prepareShowInference(
    model,
    engine=None,
    evs=None,
    targets=None,
    size=None,
    nodeColor=None,
    factorColor=None,
    arcWidth=None,
    arcColor=None,
    cmap=None,
    cmapArc=None,
    graph=None,
    view=None,
):
    """
    Transform an inference for a model in a dot representation

    :param model:
    :param engine:
    :param evs:
    :param targets:
    :param size:
    :param nodeColor:
    :param factorColor:
    :param arcWidth:
    :param arcColor:
    :param cmap:
    :param cmapArc:
    :param graph:
    :param view:
    :return:
    """
    if size is None:
        size = gum.config["notebook", "default_graph_inference_size"]

    if evs is None:
        evs = {}

    if targets is None:
        targets = {}

    if isinstance(model, gum.BayesNet):
        if engine is None:
            engine = gum.LazyPropagation(model)
        return BNinference2dot(
            model,
            size=size,
            engine=engine,
            evs=evs,
            targets=targets,
            nodeColor=nodeColor,
            arcWidth=arcWidth,
            arcColor=arcColor,
            cmapNode=cmap,
            cmapArc=cmapArc,
        )
    if isinstance(model, gum.MarkovNet):
        if view is None:
            view = gum.config["notebook", "default_markovnetwork_view"]
        if engine is None:
            engine = gum.ShaferShenoyMNInference(model)

        if view == "graph":
            return MNinference2UGdot(
                model,
                size=size,
                engine=engine,
                evs=evs,
                targets=targets,
                nodeColor=nodeColor,
                factorColor=factorColor,
                arcWidth=arcWidth,
                arcColor=arcColor,
                cmapNode=cmap,
                cmapArc=cmapArc,
            )
        # view=factor graph
        return MNinference2FactorGraphdot(
            model,
            size=size,
            engine=engine,
            evs=evs,
            targets=targets,
            nodeColor=nodeColor,
            factorColor=factorColor,
            cmapNode=cmap,
        )
    if isinstance(model, gum.InfluenceDiagram):
        if engine is None:
            engine = gum.ShaferShenoyLIMIDInference(model)
        return LIMIDinference2dot(
            model, size=size, engine=engine, evs=evs, targets=targets
        )
    if isinstance(model, gum.CredalNet):
        if engine is None:
            engine = gum.CNMonteCarloSampling(model)
        return CNinference2dot(
            model,
            size=size,
            engine=engine,
            evs=evs,
            targets=targets,
            nodeColor=nodeColor,
            arcWidth=arcWidth,
            arcColor=arcColor,
            cmapNode=cmap,
        )

    raise gum.InvalidArgument(
        "Argument model should be a PGM (BayesNet, MarkovNet or Influence Diagram"
    )


def prepareLinksForSVG(mainSvg):
    """
    Inlining links in svg

    :param mainSvg:
    :return: the new version with inlined links
    """
    re_images = re.compile(r"(<image [^>]*>)")
    re_xlink = re.compile(r"xlink:href=\"([^\"]*)")
    re_viewbox = re.compile(r"(viewBox=\"[^\"]*\")")

    # analyze mainSvg (find the secondary svgs)
    __fragments = {}
    for img in re.finditer(re_images, mainSvg):
        # print(img)
        secondarySvg = re.findall(re_xlink, img.group(1))[0]
        content = ""
        with open(secondarySvg, encoding="utf8") as f:
            inSvg = False
            for line in f:
                if line[0:4] == "<svg":
                    inSvg = True
                    viewBox = re.findall(re_viewbox, line)[0]
                    # print("VIEWBOX {}".format(viewBox))
                elif inSvg:
                    content += line
        __fragments[secondarySvg] = (viewBox, content)

    if len(__fragments) > 0:
        # replace image tags by svg tags
        img2svg = re.sub(r"<image ([^>]*)/>", "<svg \g<1>>", mainSvg)

        # insert secondaries into main
        def ___insertSecondarySvgs(matchObj):
            vb, code = __fragments[matchObj.group(1)]
            return vb + matchObj.group(2) + code

        mainSvg = re.sub(r'xlink:href="([^"]*)"(.*>)', ___insertSecondarySvgs, img2svg)

    # remove buggy white-space (for notebooks)
    mainSvg = mainSvg.replace("white-space:pre;", "")
    return mainSvg


def dot_as_svg_string(gr, size):
    """
    repr a pydot graph in a notebook

    :param string size : size of the rendered graph
    """
    if size is not None:
        gr.set_size(size)

    gsvg = prepareLinksForSVG(gr.create_svg(encoding="utf-8").decode("utf-8"))
    return gsvg


def exportInference(model, filename=None, **kwargs):
    """
    the graphical representation of an inference in a notebook

    :param GraphicalModel model: the model in which to infer (pyAgrum.BayesNet, pyAgrum.MarkovNet or
            pyAgrum.InfluenceDiagram)
    :param str filename: the name of the resulting file (suffix in ['pdf', 'png', 'ps']). If filename is None, the result is a np.array ready to be used with imshow().
    :param gum.Inference engine: inference algorithm used. If None, gum.LazyPropagation will be used for BayesNet,
            gum.ShaferShenoy for gum.MarkovNet and gum.ShaferShenoyLIMIDInference for gum.InfluenceDiagram.
    :param dictionnary evs: map of evidence
    :param set targets: set of targets
    :param string size: size of the rendered graph
    :param nodeColor: a nodeMap of values (between 0 and 1) to be shown as color of nodes (with special colors for 0 and 1)
    :param factorColor: a nodeMap of values (between 0 and 1) to be shown as color of factors (in MarkovNet representation)
    :param arcWidth: a arcMap of values to be shown as width of arcs
    :param arcColor: a arcMap of values (between 0 and 1) to be shown as color of arcs
    :param cmap: color map to show the color of nodes and arcs
    :param cmapArc: color map to show the vals of Arcs.
    :param graph: only shows nodes that have their id in the graph (and not in the whole BN)
    :param view: graph | factorgraph | None (default) for Markov network
    :return: the desired representation of the inference
    """
    if filename is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        exportInference(model, tmp.name, **kwargs)
        img = mpimg.imread(tmp.name)
        try:
            os.remove(tmp.name)
        except (
            PermissionError
        ):  # probably windows error : file still 'used' ... grrr...
            pass
        return img

    fmt_image = filename.split(".")[-1]
    if fmt_image not in ["pdf", "png", "ps"]:
        raise Exception(
            f"{filename} in not a correct filename for export : extension '{fmt_image}' not in [pdf,png,ps]."
        )

    import cairosvg

    if "size" in kwargs:
        size = kwargs["size"]
    else:
        size = gum.config["notebook", "default_graph_inference_size"]

    svgtxt = dot_as_svg_string(prepareShowInference(model, **kwargs), size=size)

    if fmt_image == "pdf":
        cairosvg.svg2pdf(bytestring=svgtxt, write_to=filename)
    elif fmt_image == "png":
        cairosvg.svg2png(bytestring=svgtxt, write_to=filename)
    else:  # format=="ps"
        cairosvg.svg2ps(bytestring=svgtxt, write_to=filename)
