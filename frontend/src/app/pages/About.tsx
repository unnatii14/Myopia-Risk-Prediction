import { motion } from "motion/react";
import { Link } from "react-router";
import {
  ExternalLink, BookOpen, FlaskConical, Target, Users,
  TrendingUp, Award, ChevronRight, Database, Brain
} from "lucide-react";

const researchers = [
  {
    institution: "LVPEI — L V Prasad Eye Institute",
    description: "India's premier eye research institution. Mission Myopia is LVPEI's nationwide programme to address childhood myopia through screening, education, and research.",
    url: "https://missionmyopia.lvpei.org/",
    badge: "Clinical Partner",
    color: "var(--primary-green)",
  },
  {
    institution: "PREMo — Pre-Myopia Risk Score",
    description: "Novel scoring system to detect pre-myopia before it occurs, enabling early intervention. Developed to identify children in the critical window before refractive changes begin.",
    url: "https://myopiaonset.com/",
    badge: "Validation Reference",
    color: "var(--secondary-green)",
  },
  {
    institution: "MPRAS — Myopia Prediction Risk Assessment Score",
    description: "Published in Nature Scientific Reports (2023). A validated risk score for myopia onset and progression, used as benchmark to compare MyopiaGuard's model performance.",
    url: "https://www.nature.com/articles/s41598-023-35696-2",
    badge: "Benchmarked Against",
    color: "var(--accent-blue)",
  },
  {
    institution: "BHVI — Brien Holden Vision Institute",
    description: "Global authority on myopia management. Their myopia progression calculator and clinical guidelines inform our risk thresholds and treatment recommendation logic.",
    url: "https://bhvi.org/myopia-calculator-resources/",
    badge: "Guidelines Reference",
    color: "var(--moderate-risk)",
  },
];

const papers = [
  {
    title: "Prediction of Refractive Error and Its Progression: A Machine Learning Algorithm",
    authors: "Barraza-Bernal et al.",
    journal: "BMJ Open Ophthalmology, 2023",
    description: "Population data from 12,780 Chinese children used to develop SVR + Gaussian Process Regression for predicting refractive error progression. Informed our Stage 3 diopter estimation model.",
    url: "https://bmjophth.bmj.com/content/8/1/e001298",
  },
  {
    title: "Prediction of Myopia in Adolescents through Machine Learning Methods",
    authors: "Yang et al., Beijing Institute of Technology",
    journal: "International Journal of Environmental Research and Public Health, 2020",
    description: "SVM-based model using reading/writing posture, eye habits, and heredity to predict myopia. Validated our feature selection approach, particularly near work and screen time.",
    url: "https://www.mdpi.com/1660-4601/17/2/463",
  },
  {
    title: "Application of Big-Data for Epidemiological Studies of Refractive Error",
    authors: "Moore, Loughman, Butler et al. — Technological University Dublin",
    journal: "PLOS ONE, 2021",
    description: "Analysis of 555,528 patient visits and 141 million spectacle lens records to understand the distribution of refractive error at a population level. Informs our severity categorisation.",
    url: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250468",
  },
  {
    title: "Open Fundus Photograph Dataset with Pathologic Myopia Recognition",
    authors: "Various — Nature Scientific Data, 2024",
    journal: "Nature Scientific Data, 2024",
    description: "Curated fundus image dataset with pathologic myopia annotations. Referenced for future Phase 3 work integrating retinal image analysis into the risk pipeline.",
    url: "https://www.nature.com/articles/s41597-024-02911-2",
  },
];

const modelMetrics = [
  { label: "Dataset Size", value: "5,000", sub: "pediatric records" },
  { label: "Training Split", value: "80%", sub: "4,000 samples" },
  { label: "Model Type", value: "XGBoost", sub: "Stage 2 classifier" },
  { label: "ROC-AUC", value: "0.88", sub: "Stage 2 risk model" },
  { label: "Accuracy", value: "81.2%", sub: "on held-out test set" },
  { label: "Features", value: "35", sub: "leakage-free inputs" },
];

const pipeline = [
  {
    stage: "Stage 1",
    title: "Refractive Error Screening",
    model: "XGBoost Classifier",
    metric: "AUC: 0.50",
    description: "Detects likelihood of any refractive error presence. Currently limited by absence of clinical visual acuity data — improves significantly when combined with optometric records.",
    status: "approximate",
  },
  {
    stage: "Stage 2",
    title: "Progression Risk Classification",
    model: "XGBoost Classifier",
    metric: "AUC: 0.88 ✦",
    description: "Primary production model. Classifies children as High Risk vs Low/Moderate Risk for myopia progression. Trained on 35 lifestyle, demographic, and genetic features with full data-leakage prevention.",
    status: "production",
  },
  {
    stage: "Stage 3",
    title: "Diopter Severity Estimation",
    model: "Gradient Boosting Regressor",
    metric: "MAE: 1.75 D",
    description: "Estimates approximate diopter severity for children identified as RE-positive. Treat as indicative only — accurate severity requires autorefraction measurement.",
    status: "approximate",
  },
];

export default function About() {
  return (
    <div className="w-full">
      {/* HERO */}
      <section className="bg-gradient-to-br from-[var(--primary-green)] to-[var(--secondary-green)] text-white py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="max-w-3xl"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-sm rounded-full mb-6 border border-white/30">
              <BookOpen className="w-4 h-4" />
              <span className="text-sm font-medium">Research & Methodology</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
              Built on Evidence.<br />Designed for India.
            </h1>
            <p className="text-xl text-white/90 leading-relaxed mb-8">
              MyopiaGuard is an AI-based myopia risk prediction model developed for school-aged
              children in India — combining clinical research, machine learning, and public health
              goals into a free community screening tool.
            </p>
            <Link
              to="/screen"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white text-[var(--primary-green)] rounded-full font-semibold hover:bg-gray-50 transition-all shadow-lg hover:scale-105 transform"
            >
              Start Free Screening <ChevronRight className="w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* PROJECT AIM */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold text-[var(--text-dark)] mb-6">
                Why This Project Exists
              </h2>
              <div className="space-y-5 text-[var(--text-muted)] leading-relaxed text-lg">
                <p>
                  Myopia has emerged as a major public health concern worldwide, with rising prevalence
                  attributed to increased near-work activities, excessive screen exposure, and reduced
                  outdoor time. Early onset is associated with faster progression and a higher risk of
                  sight-threatening complications later in life.
                </p>
                <p>
                  Conventional school vision screening primarily detects <em>established</em> myopia and
                  often fails to identify children at risk <em>before</em> refractive changes occur.
                  AI-based risk prediction enables a shift from reactive detection to proactive prevention
                  — enabling timely interventions at the community level.
                </p>
                <p>
                  There is a clear research gap: limited studies have translated AI-based prediction into
                  tools suitable for community and school settings, especially in low- and middle-income
                  settings like India.
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="grid grid-cols-2 gap-4"
            >
              {[
                { icon: Target, title: "Aim", text: "Identify high-risk children using lifestyle + genetic factors before myopia establishes" },
                { icon: Users, title: "Target", text: "School-aged children aged 5–18 in India, screened by parents or health workers" },
                { icon: TrendingUp, title: "Outcome", text: "Interpretable risk score for integration into school and community vision programmes" },
                { icon: Award, title: "Standard", text: "Validated against LVPEI clinical data, benchmarked against MPRAS risk score" },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="bg-[var(--background-mint)] rounded-2xl p-6 border border-[var(--border)]"
                >
                  <div className="w-10 h-10 bg-[var(--secondary-green)]/20 rounded-xl flex items-center justify-center mb-4">
                    <item.icon className="w-5 h-5 text-[var(--primary-green)]" />
                  </div>
                  <h4 className="font-bold text-[var(--text-dark)] mb-2">{item.title}</h4>
                  <p className="text-sm text-[var(--text-muted)] leading-relaxed">{item.text}</p>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* MODEL METRICS */}
      <section className="py-20 bg-[var(--background-mint)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full mb-4 border border-[var(--border)]">
              <Database className="w-4 h-4 text-[var(--primary-green)]" />
              <span className="text-sm font-medium text-[var(--text-dark)]">Model Statistics</span>
            </div>
            <h2 className="text-4xl font-bold text-[var(--text-dark)] mb-4">
              The Numbers Behind the Score
            </h2>
            <p className="text-xl text-[var(--text-muted)] max-w-2xl mx-auto">
              Trained on 5,000 Indian pediatric records with rigorous data-leakage prevention
            </p>
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-16">
            {modelMetrics.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="bg-white rounded-2xl p-6 text-center shadow-sm border border-[var(--border)]"
              >
                <div className="text-3xl font-bold text-[var(--primary-green)] mb-1">{m.value}</div>
                <div className="text-sm font-semibold text-[var(--text-dark)] mb-1">{m.label}</div>
                <div className="text-xs text-[var(--text-muted)]">{m.sub}</div>
              </motion.div>
            ))}
          </div>

          {/* Three-Stage Pipeline */}
          <h3 className="text-3xl font-bold text-[var(--text-dark)] mb-8 text-center">Three-Stage Pipeline</h3>
          <div className="grid md:grid-cols-3 gap-6">
            {pipeline.map((p, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.15 }}
                className={`bg-white rounded-3xl p-8 border-2 ${
                  p.status === "production"
                    ? "border-[var(--primary-green)]"
                    : "border-[var(--border)]"
                } relative overflow-hidden shadow-sm`}
              >
                {p.status === "production" && (
                  <div className="absolute top-4 right-4 px-3 py-1 bg-[var(--primary-green)] text-white text-xs font-bold rounded-full">
                    Production Ready
                  </div>
                )}
                {p.status === "approximate" && (
                  <div className="absolute top-4 right-4 px-3 py-1 bg-[var(--moderate-risk)]/20 text-[var(--moderate-risk)] text-xs font-bold rounded-full">
                    Approximate
                  </div>
                )}
                <div className="text-sm font-bold text-[var(--text-muted)] mb-2">{p.stage}</div>
                <h4 className="text-xl font-bold text-[var(--text-dark)] mb-1">{p.title}</h4>
                <div className="text-sm text-[var(--secondary-green)] font-medium mb-1">{p.model}</div>
                <div className="text-2xl font-bold text-[var(--primary-green)] mb-4">{p.metric}</div>
                <p className="text-sm text-[var(--text-muted)] leading-relaxed">{p.description}</p>
              </motion.div>
            ))}
          </div>

          <div className="mt-8 p-6 bg-white rounded-2xl border border-[var(--border)]">
            <div className="flex items-start gap-3">
              <Brain className="w-5 h-5 text-[var(--accent-blue)] mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-semibold text-[var(--text-dark)] mb-1">Data Leakage Prevention</p>
                <p className="text-sm text-[var(--text-muted)]">
                  <code className="bg-gray-100 px-1 rounded text-xs">Degree_RE_Diopters</code>,{" "}
                  <code className="bg-gray-100 px-1 rounded text-xs">Type_of_RE</code>,{" "}
                  <code className="bg-gray-100 px-1 rounded text-xs">Correction_Method</code>, and{" "}
                  <code className="bg-gray-100 px-1 rounded text-xs">Diagnosis_Age</code> were removed
                  from all model features. These are post-diagnosis measurements unavailable during
                  pre-screening — including them would artificially inflate AUC and produce a model useless
                  in the real world.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* PARTNER INSTITUTIONS */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-[var(--text-dark)] mb-4">
              Research Backing
            </h2>
            <p className="text-xl text-[var(--text-muted)] max-w-2xl mx-auto">
              Built on the work of leading ophthalmic research institutions
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6 mb-16">
            {researchers.map((r, i) => (
              <motion.a
                key={i}
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="group bg-[var(--background-mint)] rounded-3xl p-8 border border-[var(--border)] hover:border-[var(--secondary-green)] hover:shadow-lg transition-all block"
              >
                <div className="flex items-start justify-between mb-4">
                  <span
                    className="inline-block px-3 py-1 rounded-full text-xs font-bold text-white"
                    style={{ backgroundColor: r.color }}
                  >
                    {r.badge}
                  </span>
                  <ExternalLink className="w-4 h-4 text-[var(--text-muted)] group-hover:text-[var(--primary-green)] transition-colors" />
                </div>
                <h3 className="text-xl font-bold text-[var(--text-dark)] mb-3 group-hover:text-[var(--primary-green)] transition-colors">
                  {r.institution}
                </h3>
                <p className="text-sm text-[var(--text-muted)] leading-relaxed">
                  {r.description}
                </p>
              </motion.a>
            ))}
          </div>

          {/* Research Papers */}
          <h3 className="text-3xl font-bold text-[var(--text-dark)] mb-8 text-center">
            Research Papers Used
          </h3>
          <div className="space-y-4">
            {papers.map((paper, i) => (
              <motion.a
                key={i}
                href={paper.url}
                target="_blank"
                rel="noopener noreferrer"
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="group flex items-start gap-6 bg-white border border-[var(--border)] rounded-2xl p-6 hover:border-[var(--secondary-green)] hover:shadow-md transition-all block"
              >
                <div className="w-10 h-10 rounded-xl bg-[var(--secondary-green)]/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <FlaskConical className="w-5 h-5 text-[var(--primary-green)]" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h4 className="font-bold text-[var(--text-dark)] mb-1 group-hover:text-[var(--primary-green)] transition-colors">
                        {paper.title}
                      </h4>
                      <p className="text-xs text-[var(--secondary-green)] font-semibold mb-2">
                        {paper.authors} · {paper.journal}
                      </p>
                      <p className="text-sm text-[var(--text-muted)] leading-relaxed">
                        {paper.description}
                      </p>
                    </div>
                    <ExternalLink className="w-4 h-4 text-[var(--text-muted)] group-hover:text-[var(--primary-green)] transition-colors flex-shrink-0 mt-1" />
                  </div>
                </div>
              </motion.a>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 bg-gradient-to-br from-[var(--primary-green)] to-[var(--secondary-green)] text-white">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold mb-4">Ready to Screen Your Child?</h2>
            <p className="text-xl text-white/90 mb-8">
              Free · Takes 3 minutes · No equipment needed
            </p>
            <Link
              to="/screen"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white text-[var(--primary-green)] rounded-full font-semibold text-lg hover:bg-gray-50 transition-all shadow-lg hover:scale-105 transform"
            >
              Start Free Screening <ChevronRight className="w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
