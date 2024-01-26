package io.leavesfly.tinydl.utils;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.func.Variable;

import java.util.Objects;

/**
 * UML绘图，绘制计算图
 */
public class Uml {

    /**
     * 在线验证：http://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000
     */
    public static String getDotGraph(Variable variableNode) {
        StringBuilder dotGraph = new StringBuilder();
        getDotNode(variableNode, dotGraph);
        return "@startuml\ndigraph g {\n" + "rankdir=LR;\n" + dotGraph + "}\n@enduml";
    }

    private static void getDotNode(Variable variableNode, StringBuilder dotGraph) {

        if (!Objects.isNull(variableNode)) {
            String dotNode = getDotVar(variableNode);
            dotGraph.append(dotNode);

            Function functionNode = variableNode.getCreator();
            if (!Objects.isNull(functionNode)) {
                dotGraph.append(getDotFunc(functionNode));

                Variable[] inputs = functionNode.getInputs();
                if (!Objects.isNull(inputs)) {
                    for (Variable input : inputs) {
                        getDotNode(input, dotGraph);
                    }
                }
            }
        }
    }

    private static String getDotVar(Variable node) {
        String dotVar = "%s [label=\"%s\", color=orange, style=filled]\n";

        StringBuilder label = new StringBuilder();
        if (node.getName() != null) {
            label.append(node.getName()).append(" ");
        }
//        label.append(" value:").append(node.getValue().getNumber().floatValue()).append(" ");
//        if (node.getGrad() != null) {
//            label.append(" grad:").append(node.getGrad().getNumber().floatValue()).append(" ");
//        }

        return String.format(dotVar, node.hashCode(), label);
    }

    private static String getDotFunc(Function node) {
        StringBuilder dotFunc = new StringBuilder("%s [label=\"%s\", color=lightblue, style=filled, shape=box]\n");
        String docEdge = " %s -> %s\n";
        dotFunc = new StringBuilder(String.format(dotFunc.toString(), node.hashCode(), node.getClass().getSimpleName()));
        for (Variable input : node.getInputs()) {
            dotFunc.append(String.format(docEdge, input.hashCode(), node.hashCode()));
        }
        dotFunc.append(String.format(docEdge, node.hashCode(), node.getOutput().hashCode()));
        return dotFunc.toString();
    }
}
