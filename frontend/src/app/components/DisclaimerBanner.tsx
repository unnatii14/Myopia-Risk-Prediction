import { AlertTriangle } from "lucide-react";

export default function DisclaimerBanner() {
  return (
    <div className="bg-yellow-50 border-b border-yellow-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5 flex items-start sm:items-center gap-2.5">
        <AlertTriangle className="w-4 h-4 text-yellow-600 flex-shrink-0 mt-0.5 sm:mt-0" />
        <p className="text-xs sm:text-sm text-[var(--text-dark)]">
          <strong>Not a medical diagnosis.</strong> This tool gives a risk
          assessment from lifestyle factors and family history using an ML
          model trained on Indian children&apos;s data. Always consult a
          qualified ophthalmologist for proper eye examination and diagnosis.
        </p>
      </div>
    </div>
  );
}
