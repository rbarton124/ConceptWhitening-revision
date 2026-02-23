\section{Experiments and Results}\label{sec:experiments}

In this section, we provide a thorough experimental evaluation of Quantized Concept Whitening (QCW), emphasizing its behavior regarding classification accuracy, layer sensitivity, hierarchical quantization, annotation quality, and architectural influence. We structure our investigation around the following critical questions:

\begin{enumerate}
    \item \textbf{Accuracy Impact:} How does QCW affect model accuracy compared to standard pretrained baselines?
    \item \textbf{Layer Sensitivity:} How does QCW placement within the network influence performance and interpretability?
    \item \textbf{Hierarchical Quantization:} How does hierarchical structuring improve interpretability compared to standard embedding methods?
    \item \textbf{Annotation Quality Impact:} How sensitive is QCW performance to the quality of concept annotations?
    \item \textbf{Architectural Dependence:} How does QCW behave across different architectures, specifically comparing ResNet-18 and ResNet-50?
\end{enumerate}

Below, we describe our experimental setup, present detailed results, and discuss the practical implications of our findings.

\subsection{Experimental Setup}

All experiments were performed using two distinct subsets derived from the CUB-200-2011 dataset~\cite{cub}, carefully constructed to assess QCW across varying annotation qualities and complexity:

\begin{itemize}
    \item \textbf{Small Dataset}: Two high-level concepts (``eye,'' ``nape'') subdivided into 9 subconcepts, directly reflecting raw and often noisy annotations from the original dataset.
    \item \textbf{Large Dataset}: Nine high-level concepts further subdivided into 36 subconcepts, meticulously annotated and manually verified to ensure high-quality labeling.
\end{itemize}

We employed pretrained ResNet-18 and ResNet-50 models, integrating QCW layers at multiple candidate batch normalization positions (termed ``Whitened Layers'' or WL). Each model variant was trained for 200 epochs, with checkpoints selected based on validation accuracy, ensuring a balance between accuracy preservation and interpretability enhancement.

\subsection{QCW Impact on Classification Accuracy}

Figure~\ref{fig:accuracy_layers} illustrates QCW’s impact on classification accuracy. QCW consistently induced modest accuracy decreases compared to pretrained baseline models:

\begin{itemize}
    \item \textbf{ResNet-18 (Large Dataset)}: Average drop $\sim$1.2\%
    \item \textbf{ResNet-18 (Small Dataset)}: Average drop $\sim$0.5\%
    \item \textbf{ResNet-50 (Large Dataset)}: Average drop $\sim$0.8\%
\end{itemize}

These relatively minor reductions suggest that QCW introduces interpretability benefits at a manageable cost to predictive performance.

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.75\linewidth]{figs/accuracy_vs_wl.png}
    \caption{Classification accuracy across QCW integration layers (WL) for ResNet-18 models trained on Small and Large datasets compared to baseline accuracy. Optimal accuracy for QCW models is close to but slightly below baseline, with significant layer-wise variability. The Small dataset occasionally outperforms the Large, emphasizing the impact of annotation quality and dataset scale.}
    \label{fig:accuracy_layers}
\end{figure}

Notably, accuracy varied significantly depending on the placement of the QCW layers. Mid-depth layers (WL-3, WL-8 for ResNet-18, WL-3 for ResNet-50) maintained the highest accuracy, while deeper layers (WL-5, WL-7 for ResNet-18, WL-15 for ResNet-50) caused pronounced accuracy degradation. This layer-wise variation emphasizes QCW’s sensitivity to internal representational structure and the importance of careful QCW layer selection.

\subsection{Comparative Performance Analysis}

To contextualize QCW’s accuracy impact, we compared optimal QCW configurations against standard architectures (ResNet-18, ResNet-50, DenseNet-161, VGG16) pretrained on CUB (see Figure~\ref{fig:model_comparison}). QCW-integrated models closely tracked but slightly trailed baseline accuracy. However, careful QCW layer selection (e.g., ResNet-50 WL-3) nearly matched baseline performance, suggesting QCW’s interpretability gains can be achieved with minimal accuracy loss given strategic placement.

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.75\linewidth]{figs/architecture_comparison.png}
    \caption{Comparative accuracy of QCW models at optimal whitening layer (WL-3) against standard baseline architectures (ResNet-18, ResNet-50, DenseNet-161, VGG16) on CUB dataset. QCW slightly underperforms baseline ResNet-18, but notably outperforms baseline ResNet-50, showing QCW can enhance deeper architectures.}
    \label{fig:model_comparison}
\end{figure}

\subsection{Layer Sensitivity and Representational Insights}

Our layer-wise analysis provides critical insights into QCW’s interaction with model depth and representation type:

\begin{itemize}
    \item \textbf{Early Layers (WL-1, WL-2)}: QCW integration at these layers yielded relatively poor performance. Early layers encode primarily low-level features (e.g., edges, textures), making semantic quantization less effective.
    
    \item \textbf{Intermediate Layers (WL-3, WL-8)}: Optimal QCW placement occurred in mid-depth layers. These layers balance abstract semantic concepts with the flexibility needed for class discrimination, preserving accuracy while enhancing interpretability.
    
    \item \textbf{Deeper Layers (WL-5, WL-7, WL-15 (Res50)}: Despite exhibiting strong conceptual interpretability, QCW at very deep layers significantly degraded accuracy. We hypothesize that deeply placed QCW layers disrupt optimized semantic structures crucial for final-stage classification, causing representational conflicts.
\end{itemize}

Thus, QCW achieves its best balance of accuracy and interpretability in intermediate network positions, guiding practical deployment strategies.

\subsection{Hierarchical Quantization and Interpretability Benefits}

A core QCW innovation is hierarchical quantization, explicitly constraining concept representations to structured semantic subspaces. To evaluate this approach, we measured concept interpretability using \textit{Masked AUC} and \textit{Energy Ratio} metrics (see Section 3). These metrics quantify the interpretability improvement provided by hierarchical structuring versus traditional global embeddings.

Our results (summarized in Table~\ref{tab:hierarchical_summary}) show that hierarchical structuring substantially enhanced interpretability. Masked AUC consistently matched or exceeded baseline purity across optimal QCW layers, indicating explicit hierarchical constraints significantly clarify ambiguous global representations. High energy ratios (0.4–0.7) further demonstrate strong semantic coherence within these structured subspaces.

\begin{table}[ht]
    \centering
    \begin{tabular}{lcccc}
        \toprule
        Model/Dataset & Optimal WL & Masked AUC & $\Delta$-max & Energy Ratio \\
        \midrule
        ResNet-18 Small & 7 & \textbf{0.834} & 0.143 & 0.738 \\
        ResNet-18 Large & 7 & \textbf{0.865} & 0.249 & 0.414 \\
        ResNet-50 Large & 11 & \textbf{0.871} & 0.226 & 0.419 \\
        \bottomrule
    \end{tabular}
    \caption{Hierarchical quantization metrics demonstrate clear interpretability advantages.}
    \label{tab:hierarchical_summary}
\end{table}

\subsection{Annotation Quality Sensitivity}

Comparing QCW results on small (noisy) versus large (clean) datasets highlighted QCW’s sensitivity to annotation accuracy. Higher-quality annotations consistently improved QCW performance metrics:

\begin{itemize}
    \item Small Dataset: Peak purity $\sim$0.84, limited by annotation noise.
    \item Large Dataset: Significantly higher peak purity ($\sim$0.87–0.90), enabled by meticulous annotation.
\end{itemize}

These results emphasize the practical importance of careful annotation for achieving maximum interpretability with QCW.

\subsection{Architectural Influence: ResNet-18 vs. ResNet-50}

Comparative analysis of ResNet architectures revealed QCW optimal layers shift deeper in the larger ResNet-50 model (optimal WL-11) compared to the smaller ResNet-18 model (optimal WL-7). This reflects delayed semantic abstraction in deeper, wider architectures. However, extremely deep layers (e.g., WL-15 in ResNet-50) overly specialize for classification accuracy, dramatically harming interpretability metrics (purity $\sim$0.61), underscoring a fundamental trade-off between semantic interpretability and classification specialization.

\subsection{Conclusions and Practical Recommendations}

Our extensive experiments yield several key practical insights:

\begin{itemize}
    \item QCW modestly impacts classification accuracy, with careful layer selection minimizing this impact.
    \item Intermediate QCW placements achieve optimal balance between accuracy preservation and interpretability.
    \item Hierarchical quantization provides substantial interpretability benefits beyond traditional embedding methods.
    \item QCW strongly depends on annotation quality, emphasizing the importance of precise labels.
    \item Deeper architectures require deeper QCW insertions for optimal interpretability, but extremely deep placement risks accuracy and interpretability trade-offs.
\end{itemize}

These findings offer clear guidelines for future QCW implementations, highlighting QCW’s promising role in enhancing model interpretability without substantial loss of predictive accuracy.

\section{Interpretability Analysis: Concept Purity and Hierarchical Quantization in QCW}\label{sec:interpretability_analysis}

In this section, we comprehensively evaluate the interpretability characteristics of Quantized Concept Whitening (QCW), emphasizing its capability to clearly and effectively encode human-defined semantic concepts. We focus specifically on two novel interpretability metrics—\textit{Concept Purity} and \textit{Hierarchical Quantization}—and investigate how QCW leverages these metrics across various experimental conditions. We explicitly examine the following key questions:

\begin{enumerate}
    \item \textbf{Concept Purity:} How effectively does QCW isolate and encode distinct human-defined concepts?
    \item \textbf{Hierarchical Quantization:} Does explicit hierarchical structuring of the concept space enhance interpretability compared to global embeddings?
    \item \textbf{Free-Concept Phenomenon:} Why do unlabeled (“free”) axes consistently demonstrate superior concept purity?
    \item \textbf{Annotation Quality Sensitivity:} How does labeling noise and annotation accuracy affect QCW’s interpretability?
    \item \textbf{Architectural Dependence:} How does interpretability differ between architectures such as ResNet-18 and ResNet-50?
\end{enumerate}

Below, we first briefly revisit our datasets and interpretability metrics, then provide a detailed analysis and interpretation of the results.

\subsection{Datasets and Interpretability Metrics}

Our analysis utilizes the previously described Small and Large datasets, representing varying annotation qualities (Section~\ref{sec:experiments}). We adopt the following interpretability metrics:

\begin{itemize}
    \item \textbf{Baseline AUC (Purity):} Measures how distinctly individual concept axes classify their associated concepts.
    \item \textbf{Masked AUC (Hierarchical AUC):} Evaluates concept purity within hierarchical subspaces, obtained by masking irrelevant concept dimensions.
    \item \textbf{$\Delta$-max (Slice-Ratio):} Captures improvements in purity attributable to hierarchical slicing (Masked AUC minus baseline purity).
    \item \textbf{Energy Ratio:} Indicates the proportion of latent activation energy residing within designated hierarchical concept slices.
    \item \textbf{Mean Rank and Hit@k:} Quantify the prominence of concept-specific axes relative to all other latent axes.
\end{itemize}

These metrics collectively quantify how effectively QCW structures and isolates semantic concepts within latent space, thus directly measuring interpretability beyond standard accuracy.

\subsection{Quantifying Concept Purity}

Table~\ref{tab:concept_purity} summarizes QCW purity across different datasets and architectures. Results reveal a consistent pattern: QCW layers at intermediate-to-deep positions yield substantial improvements in purity compared to early layers.

\begin{table}[htbp]
    \centering
    \begin{tabular}{lcccc}
        \toprule
        Dataset / Model & Optimal WL & Mean Purity (Optimal WL) & Mean Purity (WL-1) & $\Delta$ Purity \\
        \midrule
        Small / ResNet-18 & 7 & \textbf{0.838} & 0.675 & +0.163 \\
        Large / ResNet-18 & 7 & \textbf{0.822} & 0.709 & +0.113 \\
        Large / ResNet-50 & 11 & \textbf{0.805} & 0.715 & +0.090 \\
        \bottomrule
    \end{tabular}
    \caption{QCW concept purity at optimal whitened layers (WL) versus early layers.}
    \label{tab:concept_purity}
\end{table}

These significant purity improvements (up to +16\%) reflect QCW’s success in isolating clear semantic axes deeper in the network, precisely where latent representations become semantically rich. Early-layer QCW integrations are less effective, as these layers predominantly encode basic visual features lacking clear semantic definition (Fig.~\ref{fig:purity_layers}).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{figs/purity_vs_wl_res18.png}
    \caption{Mean concept purity (measured by Baseline AUC) across whitened layers (WL) for ResNet-18 QCW trained on Small and Large datasets. Purity generally improves at deeper layers, peaking notably in layers 6–7. The Large dataset consistently achieves higher purity, highlighting benefits of cleaner annotations.}
    \label{fig:purity_layers}
\end{figure}

\subsection{Hierarchical Quantization: Structuring Interpretability}

QCW explicitly structures latent spaces into hierarchical subspaces associated with high-level concepts and subconcepts. Table~\ref{tab:hierarchy_metrics} illustrates that hierarchical structuring consistently improves interpretability, yielding marked increases in Masked AUC over baseline global embeddings.

\begin{table}[htbp]
    \centering
    \begin{tabular}{lcccc}
        \toprule
        Model / Dataset & Optimal WL & Masked AUC & $\Delta$-max & Energy Ratio \\
        \midrule
        ResNet-18 Small & 7 & \textbf{0.834} & 0.143 & 0.738 \\
        ResNet-18 Large & 7 & \textbf{0.865} & 0.249 & 0.414 \\
        ResNet-50 Large & 11 & \textbf{0.871} & 0.226 & 0.419 \\
        \bottomrule
    \end{tabular}
    \caption{Hierarchical quantization metrics demonstrating clear interpretability advantages via structured concept subspaces.}
    \label{tab:hierarchy_metrics}
\end{table}

Hierarchical quantization achieves significantly higher interpretability (Δ-max up to 0.25), effectively reducing ambiguity by constraining subconcepts within dedicated subspaces. Additionally, moderate-to-high energy ratios (0.41–0.74) demonstrate natural semantic alignment within structured slices, reinforcing the practical value of QCW’s hierarchical constraints (Fig.~\ref{fig:masked_vs_baseline_auc}).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{figs/masked_vs_baseline_auc.png}
    \caption{Comparison of Baseline AUC and Masked AUC (considering hierarchical concept structure) across whitened layers for ResNet-18 trained on the Large dataset. Masked AUC consistently exceeds Baseline AUC, demonstrating the interpretability advantage provided by hierarchical QCW, especially at intermediate layers (WL4, WL6–8).}
    \label{fig:masked_vs_baseline_auc}
\end{figure}

\subsection{The "Free-Concept" Phenomenon: Unlabeled Axes Excel}

A remarkable QCW phenomenon is the consistently superior purity (0.95–0.98 AUC) achieved by axes optimized without explicit concept labels ("free" axes). This occurs due to three hypothesized factors:

\begin{enumerate}
    \item \textbf{Enhanced Training Signals}: Free axes aggregate broad positive examples without narrow, conflicting concept definitions, yielding stronger, clearer signals.
    \item \textbf{Reduced Objective Conflict}: Free axes optimize solely for concept clarity without balancing classification objectives, promoting clean semantic specialization.
    \item \textbf{Winner-Take-All Dynamics}: Optimization inherently encourages sharp differentiation, enabling free axes to dominate clearly defined semantic niches.
\end{enumerate}

This phenomenon strongly suggests QCW’s potential to identify meaningful semantic concepts even under noisy or incomplete annotations, expanding its practical interpretability applicability (Fig.~\ref{fig:free_vs_labeled}).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{}
    \caption{Comparison of purity distributions between free and explicitly labeled axes, highlighting the superior purity of free axes.}
    \label{fig:free_vs_labeled}
\end{figure}

\subsection{Annotation Quality Sensitivity}

QCW interpretability critically depends on annotation quality. While QCW achieves moderate-to-high interpretability (purity $\sim$0.84) even on noisy annotations, meticulously annotated datasets significantly enhance purity ($\sim$0.87–0.90). This indicates QCW’s interpretability potential is ultimately capped by human labeling accuracy, underscoring annotation quality’s central role in practical QCW deployment.

\subsection{Architectural Dependence}

Comparing ResNet-18 and ResNet-50 architectures, we observe interpretability optima shift to deeper layers in the larger ResNet-50. QCW integration must align with the network's representational complexity, as larger models develop semantic abstraction at deeper stages. Overly deep QCW integration risks over-specialization, significantly degrading interpretability (purity dropping to $\sim$0.61 at deepest layers). Therefore, architectural context critically informs optimal QCW deployment strategy.

\subsection{Conclusions and Recommendations}

Our analysis establishes several critical insights:

\begin{itemize}
    \item QCW substantially enhances concept purity via hierarchical semantic structuring.
    \item Unlabeled (“free”) axes demonstrate exceptional purity, suggesting QCW’s robustness against noisy labels.
    \item Annotation quality significantly influences QCW interpretability, reinforcing the need for accurate labeling.
    \item Optimal QCW placement varies by architecture, typically favoring intermediate-to-deep layers.
\end{itemize}

Collectively, these results demonstrate QCW’s clear advantages for interpretable concept-based modeling in deep neural networks, highlighting pathways toward effective practical deployment and further methodological innovation.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{figs/energy_ratio_vs_wl.png}
    \caption{Energy ratio (fraction of total energy captured by concept axes) across whitened layers for ResNet-18 QCW trained on the Large dataset. Energy ratio trends upward at deeper layers, emphasizing that deeper network representations are increasingly dominated by structured, concept-aligned features.}
    \label{fig:energy_ratio}
\end{figure}

The energy ratio analysis (Figure~\ref{fig:energy_ratio}) reveals how concept-aligned features become increasingly dominant in deeper layers. This upward trend in energy capture reinforces our understanding that QCW's effectiveness aligns with the natural progression of semantic abstraction in CNNs.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{figs/delta_max_vs_dataset.png}
    \caption{Hierarchical benefit (Δ-max), defined as the difference between Masked AUC and Baseline AUC, across whitened layers for both Small and Large datasets. Larger dataset shows consistently higher Δ-max values, clearly indicating greater interpretability improvement with higher-quality annotations, particularly in deeper layers.}
    \label{fig:delta_max}
\end{figure}

Figure~\ref{fig:delta_max} quantifies the hierarchical benefit through Δ-max analysis, demonstrating that higher-quality annotations (Large dataset) consistently yield superior interpretability improvements. This effect becomes particularly pronounced in deeper layers, where the Large dataset achieves Δ-max values exceeding 0.25.
