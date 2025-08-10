export interface AttributeSpec {
    key: string;                 // e.g., "hairColor"
    label: string;               // e.g., "Hair color"
    options: string[];           // label list (ordered)
    template: (opt: string) => string; // prompt template
    weight?: number;             // optional attribute weight in scoring
  }

// You can freely edit/add attributes here. Keep wording concise and consistent.
export const ATTRIBUTES: AttributeSpec[] = [
    {
      key: "skinTone",
      label: "Skin tone",
      options: ["very fair", "fair", "light", "medium", "tan", "brown", "dark"],
      template: (o) => `a headshot photo of a person with ${o} skin tone`,
      weight: 1.1,
    },
    {
      key: "hairColor",
      label: "Hair color",
      options: ["black", "dark brown", "brown", "blonde", "red", "gray"],
      template: (o) => `a headshot photo of a person with ${o} hair`,
    },
    {
      key: "hairStyle",
      label: "Hair style",
      options: ["straight", "wavy", "curly", "buzz cut", "short", "long"],
      template: (o) => `a headshot photo of a person with ${o} hair`,
    },
    {
      key: "facialHair",
      label: "Facial hair",
      options: ["none", "light stubble", "stubble", "mustache", "beard"],
      template: (o) => (o === "none" ? `a headshot photo of a clean-shaven person` : `a headshot photo of a person with ${o}`),
    },
    {
      key: "glasses",
      label: "Glasses",
      options: ["with glasses", "without glasses"],
      template: (o) => `a headshot photo of a person ${o}`,
      weight: 0.9,
    },
    {
      key: "jawline",
      label: "Jawline",
      options: ["round jawline", "square jawline", "sharp jawline", "oval jawline"],
      template: (o) => `a headshot photo of a person with a ${o}`,
      weight: 1.1,
    },
    {
      key: "eyeColor",
      label: "Eye color",
      options: ["brown eyes", "hazel eyes", "green eyes", "blue eyes", "gray eyes"],
      template: (o) => `a headshot photo of a person with ${o}`,
    },
  ];