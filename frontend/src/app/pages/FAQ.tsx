import { motion } from "motion/react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../components/ui/accordion";

export default function FAQ() {
  const faqs = [
    {
      question: "What is myopia and why is it increasing?",
      answer: "Myopia (nearsightedness) is a condition where distant objects appear blurry while close objects are clear. It's caused by the eye growing too long or the cornea being too curved. Myopia is increasing globally, especially in Asia, due to lifestyle changes: more near work (screens, reading), less outdoor time, and earlier start of intensive education. In India, myopia prevalence in school children has increased by 40% over the past decade."
    },
    {
      question: "How accurate is this AI tool?",
      answer: "Our AI model is based on XGBoost machine learning algorithm trained on over 5,000 Indian school children's data. It has an Area Under Curve (AUC) of 0.88, which indicates very good predictive accuracy. However, this is a screening tool, not a diagnostic tool. It assesses risk based on lifestyle and family history factors. For actual diagnosis of myopia, a comprehensive eye examination by a qualified optometrist or ophthalmologist is required."
    },
    {
      question: "What is atropine and is it safe for children?",
      answer: "Atropine is an eye drop medication that has been proven to slow myopia progression in children. Low-dose atropine (0.01%) is the gold standard for myopia control, reducing progression by 50-60% with minimal side effects. At this low concentration, it causes very little pupil dilation or near vision blur. It's been extensively studied in Asian populations and is considered very safe. However, it requires prescription and monitoring by an eye care professional."
    },
    {
      question: "At what age should my child get an eye exam?",
      answer: "The Indian Academy of Pediatrics recommends eye screening at ages 3, 5, and then annually from school entry onwards. If your child is at high risk (as indicated by this screening), has symptoms (squinting, headaches, sitting close to TV), or has both parents with myopia, earlier and more frequent examinations are recommended. High-risk children should be screened every 3-6 months to catch progression early."
    },
    {
      question: "What are myopia control glasses?",
      answer: "Myopia control glasses are specially designed spectacle lenses that correct vision while also slowing down myopia progression. They use peripheral defocus technology—the center provides clear vision while the periphery has a specific design that reduces the stimulus for eye elongation. Popular examples include MiYOSMART (with DIMS technology) and Stellest (with H.A.L.T. technology). Studies show they can slow progression by approximately 30%."
    },
    {
      question: "Why is outdoor time so important?",
      answer: "Outdoor time is the single most effective natural intervention for myopia prevention. Studies show that 2+ hours of outdoor daylight exposure daily can reduce myopia onset risk by 30-40%. The key factors are: bright natural light (even on cloudy days), distance viewing (relaxes the eye), and possibly vitamin D from sunlight. It doesn't have to be sports—walking, playing, or even reading outdoors counts, as long as it's in natural daylight."
    },
    {
      question: "Can myopia be reversed?",
      answer: "Currently, myopia cannot be permanently reversed without surgery. Once the eye has elongated, it cannot naturally shrink back. However, myopia progression can be significantly slowed with interventions like atropine drops, myopia control glasses/lenses, increased outdoor time, and reduced near work. In adults, refractive surgery (LASIK, SMILE) can correct the vision but doesn't change the elongated eye structure or reduce complications risk from high myopia."
    },
    {
      question: "Is screen time really that harmful?",
      answer: "Extended screen time contributes to myopia through multiple mechanisms: prolonged near work, reduced blinking (causing dry eyes), and displacement of outdoor time. The 20-20-20 rule helps: every 20 minutes, look at something 20 feet away for 20 seconds. Indian pediatricians recommend limiting recreational screen time to under 2 hours daily for children. Educational screen use should include breaks and ergonomic setup (screen at arm's length, proper lighting)."
    },
    {
      question: "What is high myopia and why is it dangerous?",
      answer: "High myopia (typically -6.00D or more) significantly increases the risk of serious eye conditions later in life: retinal detachment (10x risk), myopic macular degeneration, glaucoma, and cataracts. The risk increases with the degree of myopia. This is why early detection and slowing progression in childhood is so important—reducing a child's final myopia from -6.00D to -3.00D dramatically reduces their lifetime risk of vision-threatening complications."
    },
    {
      question: "Does this screening replace an eye doctor visit?",
      answer: "No. This AI screening tool provides a risk assessment to help you understand your child's likelihood of developing or progressing myopia based on lifestyle and family factors. It cannot diagnose myopia, measure refractive error, or detect other eye conditions. Regardless of the screening result, all children should have regular comprehensive eye examinations by qualified eye care professionals. Think of this tool as an early warning system that helps you decide when to seek professional care."
    },
    {
      question: "Is myopia hereditary?",
      answer: "Yes, genetics play a significant role. If one parent has myopia, the child's risk increases 3x. If both parents have myopia, the risk increases 6x. However, genetics alone don't determine the outcome—lifestyle factors (outdoor time, screen time, near work) are equally or more important. Children with no family history can still develop myopia due to environmental factors, and children with strong family history can avoid or minimize myopia with protective behaviors."
    },
    {
      question: "What is the difference between myopia control and vision correction?",
      answer: "Vision correction (regular glasses, contact lenses, or surgery) makes things clear but does nothing to slow the eye's growth or myopia progression. Myopia control interventions (atropine drops, myopia control glasses/lenses, ortho-K) both correct vision AND actively slow the progression of myopia. For children, myopia control is crucial because it aims to reduce their final degree of myopia, thereby reducing lifetime risk of serious complications."
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[var(--background-mint)] to-white py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold text-[var(--text-dark)] mb-4">
            Frequently Asked Questions
          </h1>
          <p className="text-xl text-[var(--text-muted)]">
            Everything you need to know about myopia and this screening tool
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Accordion type="single" collapsible className="space-y-4">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.05 }}
              >
                <AccordionItem 
                  value={`item-${index}`}
                  className="bg-white rounded-2xl shadow-lg overflow-hidden border border-[var(--border)] hover:shadow-xl transition-shadow"
                >
                  <AccordionTrigger className="px-6 py-5 hover:bg-[var(--background-mint)] text-left">
                    <span className="font-bold text-[var(--text-dark)] pr-4">
                      {faq.question}
                    </span>
                  </AccordionTrigger>
                  <AccordionContent className="px-6 pb-6 pt-2">
                    <p className="text-[var(--text-muted)] leading-relaxed">
                      {faq.answer}
                    </p>
                  </AccordionContent>
                </AccordionItem>
              </motion.div>
            ))}
          </Accordion>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mt-12 p-8 bg-gradient-to-br from-[var(--primary-green)] to-[var(--secondary-green)] text-white rounded-3xl text-center"
        >
          <h3 className="text-2xl font-bold mb-4">Still Have Questions?</h3>
          <p className="mb-6 text-white/90">
            Consult with a qualified ophthalmologist or optometrist for personalized advice
          </p>
          <a
            href="#"
            className="inline-block px-8 py-3 bg-white text-[var(--primary-green)] rounded-full font-bold hover:bg-gray-50 transition-colors"
          >
            Find Eye Care Professional
          </a>
        </motion.div>
      </div>
    </div>
  );
}
