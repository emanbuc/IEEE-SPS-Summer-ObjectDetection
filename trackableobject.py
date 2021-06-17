class TrackableObject:
	def __init__(self, objectID, bbox):
		self.objectID = objectID
		self.bbox = bbox

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False