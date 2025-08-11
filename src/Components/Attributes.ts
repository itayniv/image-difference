export interface AttributeSpec {
    key: string;                 // e.g., "hairColor"
    label: string;               // e.g., "Hair color"
    options: string[];           // label list (ordered)
    template: (opt: string) => string; // prompt template
  }

// You can freely edit/add attributes here. Keep wording concise and consistent.
export const ATTRIBUTES: AttributeSpec[] = [
    {
      key: "gender",
      label: "Gender",
      options: ["a man", "a woman", "a boy", "a girl"],
      template: (o) => `a headshot photo of ${o}`,
    },
    {
      key: "skinTone",
      label: "Skin tone",
      options: ["very fair", "fair", "light", "medium", "tan", "brown", "dark"],
      template: (o) => `a headshot photo of a person with ${o} skin tone`,
    },
    {
      key: "facialExpression",
      label: "Facial expression",
      options: ["happy", "sad", "angry", "surprised", "neutral", "confused", "smiling", "serious"],
      template: (o) => `a headshot photo of a person with a ${o} facial expression`,
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
      options: ["not wearing glasses", "wearing glasses"],
      template: (o) => `a headshot photo of a person who is ${o}`,
    },
    {
      key: "jawline",
      label: "Jawline",
      options: ["round jawline", "square jawline", "sharp jawline", "oval jawline"],
      template: (o) => `a headshot photo of a person with a ${o}`,
    },
    {
      key: "eyeColor",
      label: "Eye color",
      options: ["brown eyes", "hazel eyes", "green eyes", "blue eyes", "gray eyes"],
      template: (o) => `a headshot photo of a person with ${o}`,
    },
  ];